from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import time
from datetime import datetime
from typing import List
import base64

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"mnist_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

MODEL_METADATA = {
    "model_name": "MNIST Digit Classifier",
    "version": "1.0.0",
    "accuracy": 0.987,
    "trained_on": "2024-05-10"
}

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp"]
MAX_IMAGE_SIZE_MB = 5

gitnos = FastAPI(
    title="Gitnos API",
    version="1.0"
)

@gitnos.post("/api/v1/predict")
async def predict(images: List[UploadFile] = File(...)):
    results = []

    for image in images:
        image_bytes = await image.read()

        try:
            img = Image.open(BytesIO(image_bytes)).convert("L") 
        except:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {image.filename}")

        img = ImageOps.invert(img)              
        img = img.resize((28, 28))              
        x = np.array(img, dtype=np.float32) / 255.0
        x = x[np.newaxis, ..., np.newaxis]   

        pred = model.predict(x)[0]
        predicted_digit = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results.append({
            "filename": image.filename,
            "predicted_digit": predicted_digit,
            "confidence": confidence
        })

    return {"results": results}

@gitnos.get("/")
def read_root():
    return {"message": "Welcome to Gitnos API!"}

@gitnos.get("/api/v1/model/info")
def get_model_info():
    return {
        "model_name": MODEL_METADATA["model_name"],
        "version": MODEL_METADATA["version"],
        "accuracy": MODEL_METADATA["accuracy"],
        "trained_on": MODEL_METADATA["trained_on"]
    }

@gitnos.get("/api/v1/health")
def health_check():
    start_time = time.time()
    try:

        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = model.predict(dummy_input)
        status = "healthy"
    except Exception as e:
        status = f"unhealthy: {str(e)}"
    response_time = round((time.time() - start_time) * 1000, 2)  # in ms

    return {
        "status": status,
        "response_time_ms": response_time,
        "checked_at": datetime.now().isoformat()
    }

@gitnos.post("/api/v1/predict/threshold")
async def predict_with_threshold(payload: dict):
    """
    Accepts base64 image and confidence threshold, 
    returns only predictions above threshold.
    """
    image_b64 = payload.get("image")
    threshold = payload.get("threshold", 0.8)

    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing image data in base64 format")

    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255.0
    x = x[np.newaxis, ..., np.newaxis]

    pred = model.predict(x)[0]
    predicted_digit = int(np.argmax(pred))
    confidence = float(np.max(pred))

    if confidence < threshold:
        return JSONResponse(
            status_code=200,
            content={"message": f"No predictions above threshold {threshold}", "confidence": confidence}
        )

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "threshold": threshold
    }

@gitnos.get("/api/v1/formats")
def get_supported_formats():
    return {
        "supported_formats": SUPPORTED_FORMATS,
        "max_image_size_mb": MAX_IMAGE_SIZE_MB,
        "note": "Ensure images are grayscale or RGB; they are auto-converted to grayscale."
    }