import numpy as np

CHORDS = {
    "Major": (0, 4, 7),
    "Minor": (0, 3, 7),
    "Diminished": (0, 3, 6),
    "Augmented": (0, 4, 8),
    "Major 7th": (0, 4, 7, 11),
    "Minor 7th": (0, 3, 7, 10),
    "Dominant 7th": (0, 4, 7, 10),
}

PITCH_CLASSES = [
    "A", "A#", "B", "C", "C#", "D",
    "D#", "E", "F", "F#", "G", "G#"
]

def get_pitch_class(piano_key_index: int) -> int:
    return piano_key_index % 12

def get_note_name(piano_key_index: int) -> str:
    pitch_class = get_pitch_class(piano_key_index)
    return PITCH_CLASSES[pitch_class]

def predict_chord(top_notes: list) -> tuple[str, float]:

    if not top_notes:
        return "No Notes Detected", 0.0

    unique_pitch_classes = np.unique([get_pitch_class(n) for n in top_notes])

    max_score = -1.0
    best_prediction = "Unknown"

    for root_pitch_class in unique_pitch_classes:
        root_name = PITCH_CLASSES[root_pitch_class]

        for chord_name, intervals in CHORDS.items():
            current_score = 0.0
            
            for interval in intervals:
                expected_pitch_class = (root_pitch_class + interval) % 12
                
                if expected_pitch_class in unique_pitch_classes:
                    if interval in [3, 4]:    # 3rd
                        current_score += 2.5
                    elif interval in [10, 11]: # 7th
                        current_score += 2.0
                    elif interval == 7:       # 5th
                        current_score += 1.5
                    else:                     # Root
                        current_score += 1.0

            for pitch in unique_pitch_classes:
                interval_from_root = (pitch - root_pitch_class + 12) % 12
                
                if interval_from_root not in intervals:
                    if interval_from_root in [1, 6, 11]:
                        current_score -= 1.0
                    else:
                        current_score -= 0.3

            if current_score > max_score:
                max_score = current_score
                best_prediction = f"{root_name} {chord_name}"

    return best_prediction, max_score