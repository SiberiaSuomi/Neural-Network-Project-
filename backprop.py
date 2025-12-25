import numpy as np

# Sigmoid + derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deravitive(x):
    return x * (1 - x)

# Example dataset: XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Parameters
np.random.seed(42)
input_layer = 2
hidden_layer = 2
output_layer = 1
learning_rate = 0.5
epochs = 10000

# Weights init
w1 = np.random.rand(input_layer, hidden_layer)
b1 = np.zeros((1, hidden_layer))
w2 = np.random.rand(hidden_layer, output_layer)
b2 = np.zeros((1, output_layer))

for epoch in range(epochs):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

# loss
loss = np.mean((y - a2) ** 2)


# Backward pass

dz2 = (a2 - y) * sigmoid_deravitive(a2)  # output layer delta
dw2 = np.dot(a1.T, dz2)
db2 = np.sum(dz2, axis=0, keepdims=True)

dz1 = np.dot(dz2, w2.T) * sigmoid_deravitive(a1)  # hidden layer delta
dw1 = np.dot(X.T, dz1)
db1 = np.sum(dz1, axis=0)

    # Update weights
w2 -= learning_rate * dw2
b2 -= learning_rate * db2
w1 -= learning_rate * dw1
b1 -= learning_rate * db1

if epoch % 1000 == 0:
    print(f"Epoch {epoch}, Loss: {loss}")
