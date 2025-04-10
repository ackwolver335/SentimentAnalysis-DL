import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# === 1. Load Dataset ===
df = pd.read_csv("processed_data.csv",low_memory = False)  # Replace with your actual file
df.dropna(subset=["text", "label"], inplace=True)

# === 2. Map String Labels to Integers ===
label_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
df['label'] = df['label'].map(label_map)

# === 3. Preprocess Text ===
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=max_len)

# Save tokenizer for future use
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# === 4. Prepare Labels ===
y = to_categorical(df["label"], num_classes=3)

# === 5. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Define Inception Module ===
def inception_module(x, filters):
    conv1 = Conv1D(filters, 1, activation='relu', padding='same')(x)
    conv3 = Conv1D(filters, 3, activation='relu', padding='same')(x)
    conv5 = Conv1D(filters, 5, activation='relu', padding='same')(x)
    pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    output = Concatenate()([conv1, conv3, conv5, pool])
    return output

# === 7. Build the Model ===
input_layer = Input(shape=(max_len,))
x = Embedding(input_dim=max_words, output_dim=128)(input_layer)

x = inception_module(x, 64)
x = inception_module(x, 32)

x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(3, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Show model summary
model.summary()

# === 8. Train Model ===
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# === 9. Evaluate the Model ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# === 10. Confusion Matrix and Report ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
cm = confusion_matrix(y_true, y_pred_classes)

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

# === 11. Save the Model ===
model.save("sentiment_google_net.keras")