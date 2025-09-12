from fastapi import APIRouter, File, UploadFile
from typing import List
import numpy as np
import cv2
import keras

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
    Returns fire=1 or no_fire=0 for each file.
    """
    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            results.append({"filename": file.filename, "result": None})
            continue

        result = analyse_frame(frame)
        results.append({"filename": file.filename, "result": result})

    return {"detections": results}
