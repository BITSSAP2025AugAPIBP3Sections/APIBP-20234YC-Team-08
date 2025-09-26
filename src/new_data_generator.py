# save_new_mnist.py
import numpy as np
from tensorflow.keras.datasets import mnist
from pathlib import Path
from datetime import datetime

NEW_DATA_PATH = Path("data/new")
NEW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load MNIST
(x_train, y_train), _ = mnist.load_data()

# Take a random 100 images as "new data" each time
indices = np.random.choice(len(x_train), size=100, replace=False)
x_new = x_train[indices] / 255.0
y_new = y_train[indices]

# Add channel dimension
x_new = x_new[..., np.newaxis]

# Create unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = NEW_DATA_PATH / f"new_batch_{timestamp}.npz"

# Save new data
np.savez(file_name, x=x_new, y=y_new)

print(f"New data saved for CT demo: {file_name}")
