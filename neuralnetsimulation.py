import numpy as np
import matplotlib.pyplot as plt

a = int(input('Enter random seed (e.g., 0 however 0 is not recommended):'))
np.random.seed(a)

x = int(input('Enter number of data points to generate (e.g., 100):'))

# --- Data generation ---
def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

# --- Sparse target vector for Buy/Hold/Sell ---
# 0 = Buy, 1 = Hold, 2 = Sell
y_stock = np.random.randint(0, 3, size=X.shape[0])

# --- Layers ---
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def backward(self, y_true):
        samples = len(y_true)
        self.dinputs = self.output.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

def categorical_cross_entropy(y_true, y_pred):
    samples = len(y_true)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
    correct_confidences = y_pred_clipped[range(samples), y_true]
    return -np.mean(np.log(correct_confidences))

# --- Initialize network ---
layer_1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer_2 = Layer_Dense(5, 3)
softmax = Activation_Softmax()

# --- Training loop ---
learning_rate = 1
epochs = 1000
losses = []

for epoch in range(epochs):
    layer_1.forward(X)
    activation1.forward(layer_1.output)
    layer_2.forward(activation1.output)
    softmax.forward(layer_2.output)
    
    loss = categorical_cross_entropy(y_stock, softmax.output)
    losses.append(loss)    
    softmax.backward(y_stock)
    layer_2.backward(softmax.dinputs, learning_rate)
    activation1.backward(layer_2.dinputs)
    layer_1.backward(activation1.dinputs, learning_rate)
    
    if epoch % 100 == 0:
        predictions = np.argmax(softmax.output, axis=1)
        acc = np.mean(predictions == y)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")


# --- Predictions ---
predictions = np.argmax(softmax.output, axis=1)
action_map = {0:'Buy', 1:'Hold', 2:'Sell'}
predicted_actions = [action_map[p] for p in predictions[:x]]  # first 10 predictions
print("\nFirst 10 predicted actions:", predicted_actions)

# --- Plotting ---
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
plt.title("Original Spiral Data")
plt.xlabel("X1")
plt.ylabel("X2")

plt.subplot(1,2,2)
plt.scatter(activation1.output[:,0], activation1.output[:,1], c=y, cmap='viridis', edgecolor='k')
plt.title("Layer 1 ReLU Activations")
plt.xlabel("Neuron 1 output")
plt.ylabel("Neuron 2 output")

plt.show()