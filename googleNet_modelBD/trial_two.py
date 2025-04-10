import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
                                     Dense, concatenate, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import os

# -----------------------------
# 1. Data Preparation & Preprocessing
# -----------------------------

# Load dataset (ensure your CSV file path is correct)
data = pd.read_csv('processed_data.csv', low_memory=False)
print("Dataset shape:", data.shape)  # Expecting around (200000, ...)

# Ensure text column is string and extract features
data['text'] = data['text'].astype(str)
texts = data['text']
labels = data['label']
scores = data['score']

# Optional: Normalize scores if they are not in [0,1]. For example, if scores range from 1 to 5:
if scores.max() > 1:
    scores = scores / scores.max()

# Encode sentiment labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))
print("Detected classes:", label_encoder.classes_)

# Tokenize text data
vocab_size = 5000       # Adjust vocabulary size as needed
maxlen = 100            # Maximum length for each text sequence
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Split dataset (80/20 split)
x_train, x_test, y_train_label, y_test_label, y_train_score, y_test_score = train_test_split(
    padded_sequences, encoded_labels, scores, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# -----------------------------
# 2. Define Inception Module (GoogLeNet-Inspired)
# -----------------------------

def inception_module(x, filters):
    # Branch 1: 1x1 convolution
    branch1 = Conv1D(filters=filters, kernel_size=1, activation='relu', padding='same')(x)
    
    # Branch 2: 1x3 convolution
    branch3 = Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
    
    # Branch 3: 1x5 convolution
    branch5 = Conv1D(filters=filters, kernel_size=5, activation='relu', padding='same')(x)
    
    # Branch 4: Max pooling followed by 1x1 convolution
    branch_pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    branch_pool = Conv1D(filters=filters, kernel_size=1, activation='relu', padding='same')(branch_pool)
    
    # Concatenate branches along the feature dimension
    x = concatenate([branch1, branch3, branch5, branch_pool], axis=-1)
    return x

# -----------------------------
# 3. Build the Multi-Task Model
# -----------------------------

# Input layer (fixed sequence length)
input_layer = Input(shape=(maxlen,))

# Embedding layer converts tokens into dense vectors
embedding = Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen)(input_layer)

# First Inception module block
inception1 = inception_module(embedding, filters=32)
bn1 = BatchNormalization()(inception1)
drop1 = Dropout(0.25)(bn1)

# Second Inception module block (stacked)
inception2 = inception_module(drop1, filters=32)
bn2 = BatchNormalization()(inception2)
drop2 = Dropout(0.25)(bn2)

# Global pooling to aggregate features
global_pool = GlobalMaxPooling1D()(drop2)

# Dense layers for feature extraction
dense = Dense(64, activation='relu')(global_pool)
dense = Dropout(0.5)(dense)

# Two output layers:
#   1. Regression output (score) with sigmoid activation
#   2. Classification output (sentiment type) with softmax activation
score_output = Dense(1, activation='sigmoid', name='score_output')(dense)
type_output = Dense(num_classes, activation='softmax', name='type_output')(dense)

# Define and compile the model
model = Model(inputs=input_layer, outputs=[score_output, type_output])
model.compile(optimizer='adam',
              loss={'score_output': 'mean_squared_error',
                    'type_output': 'sparse_categorical_crossentropy'},
              loss_weights={'score_output': 0.5, 'type_output': 1.0},
              metrics={'score_output': 'mse',
                       'type_output': 'accuracy'})

model.summary()

# -----------------------------
# 4. Set Up Callbacks & Train the Model
# -----------------------------

# Directory to save the best model
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# EarlyStopping to stop training if validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ModelCheckpoint to save the best model based on validation loss
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'best_model.h5'),
                             monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(x_train, 
                    {'score_output': y_train_score, 'type_output': y_train_label},
                    epochs=10,           # Increase epochs if needed
                    batch_size=32,       # Adjust based on your GPU/CPU resources
                    validation_split=0.2,
                    callbacks=[early_stop, checkpoint])

# -----------------------------
# 5. Evaluate the Model
# -----------------------------

results = model.evaluate(x_test, {'score_output': y_test_score, 'type_output': y_test_label})
print("\nEvaluation Results:")
print("Test Loss (Score):", results[1])
print("Test Loss (Type):", results[2])
print("Test Accuracy (Type):", results[4]*100, "%")

# -----------------------------
# 6. Confusion Matrix and Classification Report
# -----------------------------

# Get classification predictions
predictions = model.predict(x_test)[1]  # index 1 is the classification head
predicted_classes = np.argmax(predictions, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_label, predicted_classes))

print("\nClassification Report:")
print(classification_report(y_test_label, predicted_classes, target_names=label_encoder.classes_))

# -----------------------------
# 7. Sample Prediction
# -----------------------------

sample_text = "The Product was absolutely fantastic"
sample_seq = tokenizer.texts_to_sequences([sample_text])
sample_pad = pad_sequences(sample_seq, maxlen=maxlen)
pred_score, pred_type = model.predict(sample_pad)

# For regression head: use a threshold (e.g., 0.5) to determine sentiment polarity
threshold = 0.5
sentiment_score = pred_score[0][0]
sentiment_from_score = "POSITIVE" if sentiment_score >= threshold else "NEGATIVE"

# For classification head: decode the predicted class label
sentiment_type_index = np.argmax(pred_type[0])
sentiment_type = label_encoder.inverse_transform([sentiment_type_index])[0]

print("\nSample Prediction:")
print("Sentiment Score (Regression):", round(sentiment_score, 2))
print("Sentiment from Score (using threshold):", sentiment_from_score)
print("Sentiment Type (Classification):", sentiment_type)