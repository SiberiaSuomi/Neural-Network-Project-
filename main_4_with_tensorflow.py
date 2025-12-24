import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# ----- 1. Sample data -----
texts = [
    "We need more government control",
    "Free markets are the best",
    "Everyone should share resources equally",
    "Strong nationalism is key",
    "Individual freedom is sacred"
]

# Labels: 0 = Communist, 1 = Capitalist, 2 = Socialist, 3 = Nationalist, 4 = Libertarian
labels = [0, 1, 2, 3, 4]

# ----- 2. Text -> Numbers -----
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, padding="post")

# ----- 3. Labels -> One-hot -----
y = tf.keras.utils.to_categorical(labels, num_classes=5)

# ----- 4. Build Model -----
model = models.Sequential([
    layers.Embedding(input_dim=1000, output_dim=16, input_length=X.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation="relu"),
    layers.Dense(5, activation="softmax")
])

# ----- 5. Compile -----
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ----- 6. Train -----
model.fit(X, y, epochs=100, verbose=0)  # silent training

# ----- 7. Predict -----
test_texts = ["I believe everyone should have equal wealth"]
test_seq = tokenizer.texts_to_sequences(test_texts)
test_X = pad_sequences(test_seq, maxlen=X.shape[1], padding="post")

pred = model.predict(test_X)
print("Predicted ideology percentages:", pred[0])
