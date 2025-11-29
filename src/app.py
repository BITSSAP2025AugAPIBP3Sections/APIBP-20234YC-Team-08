import os
import sys
import logging
import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Basic logger configuration for ELK ingestion
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("mnist-app")

# Limit TensorFlow threading to avoid mutex crash on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

MODEL_PATH = Path("model/mnist_model.keras")
logger.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")

@st.cache_resource
def load_mnist_model():
    logger.info("Loading MNIST model")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info("Model loaded successfully")
    return model

model = load_mnist_model()

st.title("MNIST Digit Recognition Demo")

# Drawing canvas
st.write("Draw a digit (0-9) below:")
canvas = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
)

if st.button("Predict") and canvas.image_data is not None:
    logger.info("Prediction triggered by user")

    try:
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))

        x = np.array(img, dtype=np.float32) / 255.0
        x = x[np.newaxis, ..., np.newaxis]

        logger.info("Running model prediction")
        pred = model.predict(x)
        digit = int(np.argmax(pred[0]))

        logger.info(f"Predicted digit: {digit}")
        st.write(f"Predicted digit: {digit}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.write("Prediction failed. Check logs.")