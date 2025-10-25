from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"mnist_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

gitnos = FastAPI(
    title="Gitnos API",
    version="1.0"
)

@gitnos.post("/api/v1/predict")
async def predict(images: List[UploadFile] = File(...)):
    results = []

    for image in images:
        # Read file bytes
        image_bytes = await image.read()
        np_img = np.frombuffer(image_bytes, np.uint8)

        # Decode with OpenCV
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)  # ✅ Convert to grayscale

        if img is None:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {image.filename}")

        # Preprocess to MNIST format
        img = cv2.bitwise_not(img)                                      # ✅ invert colors (MNIST style)
        img = cv2.resize(img, (28, 28))                                 # ✅ Resize to 28x28
        x = img.astype(np.float32) / 255.0
        x = x[np.newaxis, ..., np.newaxis]                              # ✅ Shape: (1, 28, 28, 1)

        # Predict
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