import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Load your dataset
df = pd.read_csv('processed_data.csv')  # Ensure it has 'text' and 'label' columns

# Preprocess labels
labels = df['label'].astype('category')
df['label_encoded'] = labels.cat.codes
label_map = dict(enumerate(labels.cat.categories))

# Tokenization
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=max_len)

# Labels
y = to_categorical(df['label_encoded'])

# Split
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

# ResNet-like block
def residual_block(x, filters, kernel_size):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Add()([x, shortcut])  # Residual connection
    return x

# Build Model
inputs = Input(shape=(max_len,))
x = Embedding(max_words, 128)(inputs)
x = Conv1D(64, 5, padding='same', activation='relu')(x)
x = residual_block(x, 64, 5)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
                    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {accuracy:.2f}')

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_map.values()))

# Save Model
model.save('sentiment_resnet_model.keras')
print("✅ Model saved as 'sentiment_resnet_model.keras'")