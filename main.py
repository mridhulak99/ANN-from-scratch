from ann import ANN
import numpy as np

layers = [
    {'neurons': 10, 'activation': 'relu', 'weight': 'random'},
    {'neurons': 5, 'activation': 'relu', 'weight': 'random'},
    {'neurons': 3, 'activation': 'relu', 'weight': 'random'},
    {'neurons': 1, 'activation': 'sigmoid', 'weight': 'random'},
]
x = np.array([
    [1, 1, 0.2, 1, 1],
    [1, 0.9, 1, 0/.5, 1],
    [1, 0.6, 0.5, 1, 0.2],
    [0.5, 0.5, -0.5, 0.2, -0.1],
    [0.5, 0.5, -0.1, 0.2, -0.1],
    [0.5, 0.4, -0.5, 0.2, -0.1],
    [0.5, 0.1, -0.5, 0.2, -0.1],
])
y = np.array([[1], [1], [1], [0], [0], [0], [0]])

model = ANN(5,1)
model.initialize(layers)
model.compile(optimizerMethod="stochasticGD", lossFunction="mse", metrics=["accuracy", "precision"])
model.summary()
model.fit(x, y, epochs=100)
model.evaluate(x, y)
