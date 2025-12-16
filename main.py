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

# ------------------- FastAPI App -------------------
app = FastAPI(title="Piano Chord Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Load Models -------------------
providers = ["CPUExecutionProvider"]
model_path = os.environ.get("ONNX_MODEL_PATH", "model/final_model.onnx")
ort_session = ort.InferenceSession(model_path, providers=providers)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID")

if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
    raise ValueError("ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID must be set in environment variables")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# ------------------- Helper Functions -------------------
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

def build_chord_list(idxs, ort_output):
    chords = []
    for idx in idxs:
        chord, score = predict_chord([idx])
        chords.append({
            "note_index": int(idx),
            "note_name": get_note_name(idx),
            "chord": chord,
            "score": float(score)
        })
    return chords

# ------------------- Prediction Endpoint -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        img_bytes = await file.read()
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect keyboard
        results = client.infer(img_bgr, model_id=ROBOFLOW_MODEL_ID)
        dets = sv.Detections.from_inference(results)
        dets = dets[dets.confidence > 0.95]

        if len(dets) == 0:
            return JSONResponse(content={"error": "No keyboard detected"}, status_code=200)

        # Crop detected keyboard
        x1, y1, x2, y2 = dets.xyxy[0].astype(int)
        h, w = img.shape[:2]
        EXTENSION_PIXELS = 50
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2 + EXTENSION_PIXELS), min(h, y2)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            return JSONResponse(content={"error": "Detected box is empty"}, status_code=200)

        # Draw bounding box
        boxed_img = img_bgr.copy()
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Predict chord
        crop_input = preprocess_image(crop)
        output = ort_session.run(None, {"input": crop_input})[0][0]

        sorted_idxs = np.argsort(output)[::-1]
        top_3 = sorted_idxs[:3]
        lesser_7 = sorted_idxs[3:10]

        return {
            "images": {
                "boxed": encode_image(boxed_img),
                "cropped": encode_image(crop_bgr)
            },
            "predictions": {
                "top_3": build_chord_list(top_3, output),
                "lesser_7": build_chord_list(lesser_7, output)
            }
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
