import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Parameters
max_words = 10000  # Same as used during training
dataset_path = 'processed_data.csv'  # Replace with your actual dataset path

# Load dataset
df = pd.read_csv(dataset_path)

# Fit tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])

# Save tokenizer to a file
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer has been saved as 'tokenizer.pkl'")