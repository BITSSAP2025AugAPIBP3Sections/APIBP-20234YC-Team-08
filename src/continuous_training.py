import os
import numpy as np
from pathlib import Path
from mnist_demo import build_model
import tensorflow as tf

MODEL_PATH = Path("model/mnist_model.keras")
NEW_DATA_PATH = Path("data/new/")

def load_new_data():
    # Example: assume each file in NEW_DATA_PATH is a npz with x and y
    x_list, y_list = [], []
    for file in NEW_DATA_PATH.glob("*.npz"):
        data = np.load(file, allow_pickle=True)
        x_list.append(data['x'])
        y_list.append(data['y'])
    if x_list:
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        return x, y
    return None, None

def continuous_training():
    # Load new data
    x_new, y_new = load_new_data()
    if x_new is None:
        print("No new data found. Skipping retraining.")
        return

    # Load existing model
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded existing model.")
    else:
        model = build_model()
        print("Created new model.")

    # Retrain on new data
    model.fit(x_new, y_new, epochs=1, validation_split=0.1)
    model.save(MODEL_PATH)
    print("Model updated and saved.")

if __name__ == "__main__":
    continuous_training()
