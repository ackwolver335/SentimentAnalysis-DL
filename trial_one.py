import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset (ensure your CSV has the expected number of rows)
data = pd.read_csv('processed_data.csv', low_memory=False)
print("Dataset shape:", data.shape)  # Should be around (200000, ...)

# Extract relevant columns (ensure texts are strings)
texts = data['text'].astype(str)
labels = data['label']
scores = data['score']

# Encode sentiment labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Tokenize text data
vocab_size = 5000       # Adjust vocabulary size as needed
maxlen = 100            # Maximum length of each text sequence
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Split the dataset into training and test sets (80/20 split)
x_train, x_test, y_train_label, y_test_label, y_train_score, y_test_score = train_test_split(
    padded_sequences, encoded_labels, scores, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# Build the multi-task model (ResNet-inspired architecture)

# Input layer (each sample has length = maxlen)
input_layer = Input(shape=(maxlen,))

# Embedding layer converts words to dense vectors
embedding = Embedding(input_dim=vocab_size, output_dim=128)(input_layer)

# Convolutional block
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(embedding)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)

# Projection layer for residual connection (1x1 convolution)
projected = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(embedding)

# Residual connection
residual = add([projected, conv2])

# Global pooling and dense layers
global_pool = GlobalMaxPooling1D()(residual)
dense = Dense(64, activation='relu')(global_pool)

# Two output layers:
#   - score_output: regression (using sigmoid)
#   - type_output: classification (using softmax)
score_output = Dense(1, activation='sigmoid', name='score_output')(dense)
type_output = Dense(3, activation='softmax', name='type_output')(dense)

# Define and compile the model with appropriate losses and metrics
model = Model(inputs=input_layer, outputs=[score_output, type_output])
model.compile(optimizer='adam',
              loss={'score_output': 'mean_squared_error', 
                    'type_output': 'sparse_categorical_crossentropy'},
              metrics={'score_output': 'mse', 
                       'type_output': 'accuracy'})

model.summary()

# Train the model
history = model.fit(x_train, 
                    {'score_output': y_train_score, 'type_output': y_train_label},
                    epochs=5,
                    batch_size=32,           # Change this value to adjust the batch size
                    validation_split=0.2)

# Evaluate the model on the test set
results = model.evaluate(x_test, {'score_output': y_test_score, 'type_output': y_test_label})
print("Test Loss (Score):", results[1])
print("Test Loss (Type):", results[2])
print("Test Accuracy (Type):", results[4]*100, "%")

# Prediction on a sample text
sample_text = "The Product was absolutely fantastic"
sample_seq = tokenizer.texts_to_sequences([sample_text])
sample_pad = pad_sequences(sample_seq, maxlen=maxlen)
pred_score, pred_type = model.predict(sample_pad)

# Process predictions:
# For the regression head, use a threshold (e.g., 0.5) to determine sentiment
threshold = 0.5
sentiment_score = pred_score[0][0]
sentiment_from_score = "POSITIVE" if sentiment_score >= threshold else "NEGATIVE"

# For the classification head, take the argmax and decode it back to a label
sentiment_type_index = np.argmax(pred_type[0])
sentiment_type = label_encoder.inverse_transform([sentiment_type_index])[0]

print("Sentiment Score (Regression):", round(sentiment_score, 2))
print("Sentiment from Score (using threshold):", sentiment_from_score)
print("Sentiment Type (Classification):", sentiment_type)
