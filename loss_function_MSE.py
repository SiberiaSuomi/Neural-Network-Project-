import numpy as np

y_true = np.array([1, 0, 0, 1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# --- Create neuron ---
weights = np.array([0.5, -0.3, 0.8, 0.1])
bias = 0.0
neuron = Neuron(weights, bias)

# --- Predict ---
y_pred = neuron.feedforward(y_true)

# --- Loss ---
print("Prediction:", y_pred)
print("MSE Loss:", mse_loss(y_true, y_pred))

if y_pred >= 0.5:
    print("Value Accepted")
else:
    print("Value did not pass the activation function")