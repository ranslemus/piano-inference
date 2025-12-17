import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from inference_sdk import InferenceHTTPClient
import supervision as sv
from chord_predictor import predict_chord, get_note_name
from io import BytesIO
import os
import base64

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Piano Chord Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load ONNX model
# -------------------------
providers = ["CPUExecutionProvider"]
model_path = os.environ.get("ONNX_MODEL_PATH", "model/final_model.onnx")
ort_session = ort.InferenceSession(model_path, providers=providers)

# -------------------------
# Roboflow setup
# -------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID")

if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
    raise ValueError("ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID must be set")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# -------------------------
# Helpers
# -------------------------
def preprocess_image(crop_img):
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.resize(crop_img, (640, 480))
    crop_img = crop_img.astype(np.float32) / 255.0
    crop_img = (crop_img - 0.5) / 0.5
    crop_img = np.expand_dims(crop_img, axis=0)  # C,H,W
    crop_img = np.expand_dims(crop_img, axis=0)  # batch
    return crop_img

def encode_image(img_bgr):
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ---- Load image ----
        img_bytes = await file.read()
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ---- Detect keyboard ----
        results = client.infer(img_bgr, model_id=ROBOFLOW_MODEL_ID)
        dets = sv.Detections.from_inference(results)
        dets = dets[dets.confidence > 0.95]

        if len(dets) == 0:
            return JSONResponse(
                content={"error": "No keyboard detected"},
                status_code=200
            )

        # ---- Crop keyboard ----
        x1, y1, x2, y2 = dets.xyxy[0].astype(int)
        h, w = img.shape[:2]

        EXTENSION_PIXELS = 50
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2 + EXTENSION_PIXELS), min(h, y2)

        crop = img_bgr[y1:y2, x1:x2]

        boxed_img = img_bgr.copy()
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---- ONNX inference ----
        crop_input = preprocess_image(crop)
        output = ort_session.run(None, {"input": crop_input})[0][0]
        # output shape: (N_NOTES,)

        # ---- Top 10 notes (numeric + confidence) ----
        sorted_idxs = np.argsort(output)[::-1]
        top_10_idxs = sorted_idxs[:10]
        top_10_scores = output[top_10_idxs]

        detected_notes_top_10 = [
            {
                "note_index": int(idx),
                "note_name": get_note_name(idx),
                "confidence": float(score)
            }
            for idx, score in zip(top_10_idxs, top_10_scores)
        ]

        # ---- Predict ONE chord (NOT from top-3) ----
        predicted_chord, chord_score = predict_chord(top_10_idxs.tolist())

        # ---- Response ----
        return {
            "images": {
                "boxed": encode_image(boxed_img),
                "cropped": encode_image(crop)
            },
            "detected_notes_top_10": detected_notes_top_10,
            "predicted_chord": predicted_chord,
            "chord_score": float(chord_score)
        }

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
