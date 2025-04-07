import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model('sentiment_resnet_model.keras')

# Parameters used in training
max_len = 100  # Must match the value used during training

# Label map (use the same encoding used while training)
label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}  # Change this if your labels differ

# Function to predict sentiment
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    
    pred = model.predict(padded)[0]  # Softmax scores
    sentiment_idx = np.argmax(pred)
    sentiment_label = label_map[sentiment_idx]
    confidence_score = pred[sentiment_idx]

    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Confidence Score: {confidence_score:.2f}")
    return sentiment_label, confidence_score

# 🔍 Example usage:
text_input = "I'm extremely unhappy with how things turned out."
predict_sentiment(text_input)