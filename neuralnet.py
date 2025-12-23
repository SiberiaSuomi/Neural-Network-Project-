import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Fetch Apple stock prices ---
data = yf.download('AAPL', start='2020-01-01', end='2025-12-01')
prices = data['Close'].values

# --- 2. Compute returns ---
returns = (prices[1:] - prices[:-1]) / prices[:-1]  # daily pct change
returns = returns.reshape(-1,1)

# --- 3. Create sequences for multi-day prediction ---
seq_len = 60  # past 60 days
pred_days = 5  # predict next 5 days

def create_sequences(data, seq_len, pred_days):
    X, y = [], []
    for i in range(seq_len, len(data)-pred_days):
        X.append(data[i-seq_len:i].flatten())
        y.append(data[i:i+pred_days].flatten())
    return np.array(X), np.array(y)

X, y = create_sequences(returns, seq_len, pred_days)

# --- 4. Train/test split ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- 5. Initialize weights ---
np.random.seed(69)
input_size = seq_len
hidden_size = 50
output_size = pred_days

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

lr = 0.01

# --- 6. Activation ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- 7. Training loop ---
for epoch in range(3000):
    # forward
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = z2  # linear output for regression

    # loss
    loss = np.mean((a2 - y_train)**2)

    # backprop
    d_a2 = 2*(a2 - y_train)/y_train.size
    dW2 = a1.T @ d_a2
    db2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = d_a2 @ W2.T
    d_z1 = d_a1 * sigmoid_deriv(z1)
    dW1 = X_train.T @ d_z1
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # update
    W1 -= lr*dW1
    b1 -= lr*db1
    W2 -= lr*dW2
    b2 -= lr*db2

    if epoch % 300 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# --- 8. Predict ---
hidden = sigmoid(X_test @ W1 + b1)
pred_returns = hidden @ W2 + b2

# convert returns to price predictions
last_price = prices[split+seq_len-1]  # last actual price in training
pred_prices = []
for r in pred_returns:
    temp = []
    price = last_price
    for pct in r:
        price = price * (1 + pct)
        temp.append(price)
    last_price = temp[-1]  # next sequence starts from last predicted
    pred_prices.append(temp)
pred_prices = np.array(pred_prices)

# --- 9. Plot last prediction sequence ---
plt.figure(figsize=(14,5))
plt.plot(range(pred_days), pred_prices[-1], color='red', marker='o', label='Predicted Next 5 Days')
plt.title('Predicted Next 5 Days of AAPL Price')
plt.xlabel('Days Ahead')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
