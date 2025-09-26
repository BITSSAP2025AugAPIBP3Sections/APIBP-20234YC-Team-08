import os
import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Limit TensorFlow threading to avoid mutex crash on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

MODEL_PATH = Path("model/mnist_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

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
    # Convert canvas to 28x28 grayscale
    img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255.0
    x = x[np.newaxis, ..., np.newaxis]
    pred = model.predict(x)
    st.write(f"Predicted digit: {int(np.argmax(pred[0]))}")

port = int(os.environ.get("PORT", 8501))
st.run_server("0.0.0.0", port=port)
