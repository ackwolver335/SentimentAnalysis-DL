# importing the necessary libraries here
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Dense, Conv1D, GlobalMaxPooling1D, Flatten, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# loading the dataset regarding pre-processing
data = pd.read_csv('processed_data.csv',low_memory = False,)

# Extracting the Columns
titles = data['title']
texts = data['text']
labels = data['label']
scores = data['score']

# encoding the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# text tokenizer
tokenizer = Tokenizer(num_words = 5000)                             # vocabulary size adjustment
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen = 100)           # adjustment of Maxlen

# Splitting the Dataset
x_train,x_test, y_train_label, y_test_label, y_train_score,y_test_score = train_test_split(padded_sequences,encoded_labels, scores, test_size = 0.2, random_state = 42)

# Building the resnet inspired model
# input layers
input_layer = Input(shape=(100,))                                   # same as maxlen

# embedding layer for text to dense vector's conversion
embedding_layer = Embedding(input_dim = 5000,output_dim = 128)(input_layer)

# first convolutional block
conv1 = Conv1D(filters = 64,kernel_size = 3,activation = 'relu',padding = 'same')(embedding_layer)
conv2 = Conv1D(filters = 64,kernel_size = 3,activation = 'relu',padding = 'same')(conv1)

projected_embedding = Conv1D(filters = 64,kernel_size = 1,activation = 'relu',padding = 'same')(embedding_layer)

# Residual Connection
residual = tf.keras.layers.add([projected_embedding,conv2])

# Global Pooling
global_pool = GlobalMaxPooling1D()(residual)

# Dense Layers
dense_layer = Dense(64,activation = 'relu')(global_pool)

# Output Layers
score_output = Dense(1,activation = 'sigmoid',name = 'score_output')(dense_layer)
type_output = Dense(3,activation = 'softmax',name = 'type_output')(dense_layer)

# Building the Model
model = Model(inputs = input_layer, outputs = [score_output,type_output])
model.compile(optimizer = 'adam',loss = {'score_output' : 'mean_squared_error','type_output' : 'sparse_categorical_crossentropy'},metrics = {'score_output' : 'mse','type_output' : 'accuracy'})

model.summary()

# Training the Model
history = model.fit(x_train,{'score_output' : y_train_score,'type_output' : y_train_label},epochs = 10,batch_size = 32,validation_split = 0.2)