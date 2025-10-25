from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

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



