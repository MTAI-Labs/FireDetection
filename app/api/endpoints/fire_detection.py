from fastapi import APIRouter, File, UploadFile
from typing import List
import numpy as np
import cv2
import keras
from collections import Counter  # <-- added

router = APIRouter()

# Load model once at startup
model = keras.models.load_model("app/local/xception_final.keras")

def analyse_frame(frame: np.ndarray):
    """Analyse single frame -> fire=1, no_fire=0"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (224, 224))
    x = np.expand_dims(resized / 255.0, axis=0)

    pred = model.predict(x, verbose=0)
    fire_prob = 1 - pred[0][0]  # invert
    return int(fire_prob > 0.5)

@router.post("/detect_fire/")
async def fire_detection(files: List[UploadFile] = File(...)):
    """
    Fire detection endpoint.
    Accepts one or multiple image files.
    Returns majority vote (fire=1, no_fire=0) across all files.
    """
    predictions = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            continue  # skip invalid images

        predictions.append(analyse_frame(frame))

    if not predictions:
        return {"result": None, "message": "No valid images uploaded."}

    # Compute majority vote
    vote = Counter(predictions)
    majority_result = vote.most_common(1)[0][0]

    return {"result": majority_result, "votes": dict(vote)}
