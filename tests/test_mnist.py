import pytest
import tensorflow as tf
from src.mnist_demo import build_model

def test_model_creation():
    model = build_model()
    assert model is not None

def test_data_shape():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    assert x_train.shape[1:] == (28, 28)
    assert len(x_train) > 0
