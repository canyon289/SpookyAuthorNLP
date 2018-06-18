import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

# Setup Logging
logging.basicConfig(filename='keras_model_run.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting New Run \n")

CROSS_EVALUATE = False
FULL_EVALUATE = True
SUBMISSION = False

# Load data
from spookyauthor.data import make_dataset
train, test = make_dataset.load_raw_data()


# Preprocess  data
from spookyauthor.data.make_dataset import Preprocessing
ids, text, authors = Preprocessing.split_cols(train)

from sklearn.preprocessing import LabelBinarizer
l = LabelBinarizer()
labels = l.fit_transform(authors)

# fix random seed for reproducibility and set max words
num_words = 10000
num_input_length = 500

# Preprocess text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(text)
word_sequences = tokenizer.texts_to_sequences(text)
padded_word_sequences = pad_sequences(word_sequences, maxlen=num_input_length)


# Make model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Embedding
from keras.layers import LSTM

# 19579 rows
# 3 words
"""
Embedding needs the following parameters
input_dim = Number of vocab tokenized + 1
input_length = length of input arrays
https://stackoverflow.com/questions/40644372/keras-the-difference-input-dim-and-input-length-in-lstm
"""

model = Sequential()
model.add(Embedding(input_dim=num_words+1, output_dim=128, input_length=num_input_length))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(padded_word_sequences, labels, batch_size=500, epochs=30, validation_split=.1, shuffle=True, verbose=10)



# Get Max Prediction from each class
# subset = predictions[:10]
# print(predictions[:10])
# predicted_author = np.zeros(labels.shape)
# predicted_author[np.arange(predicted_author.shape[0]), np.argmax(predictions, axis=1)] = 1
# print(predicted_author[:10])
# print(True)

# Make Predictions
import pandas as pd
test_ids, test_text, authors = Preprocessing.split_cols(test)

word_sequences = tokenizer.texts_to_sequences(test_text)
padded_word_sequences = pad_sequences(word_sequences, maxlen=num_input_length)

predictions = model.predict(padded_word_sequences)

df = pd.DataFrame(predictions, columns=["EAP", "HPL", "MWS"])
df["id"] = test_ids
df = df.set_index("id")

df.to_csv("KerasEmbeddings.csv")