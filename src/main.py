import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Load data
print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize
print("Finished loading data")

# 2. Build a simple CNN
print("Building CNN...")
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])
print("Finished building CNN")

print("Compiling...")
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
print("Finished compiling...")

# 3. Train
print("Training...")
model.fit(x_train[..., np.newaxis], y_train, epochs=3, validation_split=0.1)
print("Finished training")

# 4. Evaluate
print("Evaluating...")
test_loss, test_acc = model.evaluate(x_test[..., np.newaxis], y_test)
print(f"Finished evaluating with test aaccuracy: {test_acc:.4f}")

# 5. Predict on a few samples
print("Predicting...")
predictions = model.predict(x_test[:5][..., np.newaxis])
for i, pred in enumerate(predictions):
    print(f"Image {i}: True label={y_test[i]}, Predicted={np.argmax(pred)}")
print("Finished predicting")

