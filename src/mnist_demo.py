# src/mnist_demo.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_model():
    """Return a compiled CNN model for MNIST."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main(epochs=3):
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build model
    model = build_model()

    # Train
    model.fit(x_train[..., np.newaxis], y_train, epochs=epochs, validation_split=0.1)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test[..., np.newaxis], y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Predict first 5 images
    predictions = model.predict(x_test[:5][..., np.newaxis])
    for i, pred in enumerate(predictions):
        print(f"Image {i}: True label={y_test[i]}, Predicted={np.argmax(pred)}")

if __name__ == "__main__":
    main()
