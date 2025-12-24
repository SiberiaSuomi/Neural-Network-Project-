import numpy as np
import matplotlib.pyplot as plt

# --- same tiny NN ---
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

class TinyNN:
    def __init__(self):
        self.W = np.random.randn(2, 4) * 0.01
        self.b = np.zeros((1, 4))
    
    def forward(self, X):
        return softmax(np.dot(X, self.W) + self.b)
    
    def compute_loss(self, X, Y):
        probs = self.forward(X)
        return -np.sum(Y * np.log(probs + 1e-8)) / X.shape[0]
    
    def train(self, X, Y, lr=0.1, epochs=1000):
        for i in range(epochs):
            probs = self.forward(X)
            grad_z = (probs - Y) / X.shape[0]
            self.W -= lr * np.dot(X.T, grad_z)
            self.b -= lr * np.sum(grad_z, axis=0, keepdims=True)
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

# --- training data ---
data = np.array([
    [0.8, 0.9],   # AR
    [-0.7, 0.6],  # AL
    [-0.5, -0.8], # LL
    [0.6, -0.7]   # LR
])
labels = np.array([0, 1, 2, 3])
def one_hot(y, num_classes=4):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh
labels_oh = one_hot(labels)

# --- train NN ---
nn = TinyNN()
nn.train(data, labels_oh)

# --- test points ---
test_points = np.array([
    [0.7, 0.8],
    [-0.6, 0.7],
    [-0.3, -0.9],
    [0.5, -0.6],
    [0.0, 0.0]
])
pred_classes, pred_probs = nn.predict(test_points)

# --- plotting ---
colors = ['red', 'blue', 'green', 'orange']
labels_map = ['Auth-Right', 'Auth-Left', 'Lib-Left', 'Lib-Right']

plt.figure(figsize=(8,8))
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# label quadrants
plt.text(0.5, 0.5, 'Auth-Right', fontsize=12, ha='center', va='center')
plt.text(-0.5, 0.5, 'Auth-Left', fontsize=12, ha='center', va='center')
plt.text(-0.5, -0.5, 'Lib-Left', fontsize=12, ha='center', va='center')
plt.text(0.5, -0.5, 'Lib-Right', fontsize=12, ha='center', va='center')

# plot training data
plt.scatter(data[:,0], data[:,1], marker='o', s=100, c='black', label='train')

# plot test points with predicted class color
for i, point in enumerate(test_points):
    plt.scatter(point[0], point[1], color=colors[pred_classes[i]], s=150, edgecolors='k')
    plt.text(point[0]+0.02, point[1]+0.02,
             f"{labels_map[pred_classes[i]]}\n{pred_probs[i].round(2)}", fontsize=8)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Economic: Left (-) → Right (+)')
plt.ylabel('Authority: Libert (-) → Auth (+)')
plt.title('Political Compass NN Predictions')
plt.grid(True)
plt.show()
