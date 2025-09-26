# save_new_mnist.py
import numpy as np
from tensorflow.keras.datasets import mnist
from pathlib import Path

NEW_DATA_PATH = Path("data/new")
NEW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load MNIST
(x_train, y_train), _ = mnist.load_data()

# Take first 100 images as "new data"
x_new = x_train[:100] / 255.0
y_new = y_train[:100]

# Save as .npz
np.savez(NEW_DATA_PATH / "new_batch_1.npz", x=x_new[..., np.newaxis], y=y_new)

print("New data saved for CT demo.")
