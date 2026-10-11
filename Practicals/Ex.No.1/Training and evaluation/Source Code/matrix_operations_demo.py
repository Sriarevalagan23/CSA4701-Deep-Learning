"""Numerical demonstration of flattening, vector operations, and a dense layer."""
import numpy as np

# Tiny 2x2 grayscale image
image = np.array([[0.2, 0.4],
                  [0.6, 0.8]], dtype=float)
x = image.flatten()  # (2,2) -> (4,)
# Four input features, three neurons. W shape is (4,3), bias shape is (3,)
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9],
              [1.0, 1.1, 1.2]])
b = np.array([0.01, 0.02, 0.03])

z = x @ W + b  # vector-matrix multiplication and bias addition
a = np.maximum(0, z)  # ReLU activation
print("Image matrix:\n", image)
print("Flattened input vector:", x)
print("Pre-activation output z = x @ W + b:", z)
print("ReLU output:", a)
print("Shapes: x", x.shape, "W", W.shape, "b", b.shape, "output", a.shape)
