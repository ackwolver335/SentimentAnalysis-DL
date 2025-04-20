import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --- Load Test Data ---
df = pd.read_csv("processed_data.csv",low_memory = False)
label_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
df['label'] = df['label'].map(label_map)
labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
X_text = df["text"].astype(str)
y_true = df["label"].values
y_cat = to_categorical(y_true, num_classes=3)

# --- Helper: Evaluate Model ---
def evaluate_model(model_path, tokenizer_path, model_name):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    sequences = tokenizer.texts_to_sequences(X_text)
    X_seq = pad_sequences(sequences, maxlen=100)

    y_pred_probs = model.predict(X_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = np.mean(y_pred == y_true) * 100
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    return {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "accuracy": acc,
        "conf_matrix": cm,
        "report": report,
        "y_pred": y_pred
    }

# --- Evaluate Both Models ---
resnet_result = evaluate_model("sentiment_resnet_model.keras", "tokenizer_resnet.pkl", "ResNet")
googlenet_result = evaluate_model("sentiment_google_net.keras", "tokenizer_googleNet.pkl", "GoogleNet")

# --- Plot Accuracy Comparison ---
model_names = [resnet_result["name"], googlenet_result["name"]]
accuracies = [resnet_result["accuracy"], googlenet_result["accuracy"]]

plt.figure(figsize=(8,5))
sns.barplot(x=model_names, y=accuracies, palette='Set2')
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.show()

# --- Plot Confusion Matrix & Print Report ---
for result in [resnet_result, googlenet_result]:
    plt.figure(figsize=(6,5))
    sns.heatmap(result["conf_matrix"], annot=True, fmt='d', cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.title(f"{result['name']} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"\n{result['name']} - Classification Report:")
    print(classification_report(y_true, result["y_pred"], target_names=labels))