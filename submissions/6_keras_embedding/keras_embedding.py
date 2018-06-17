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
id, text, authors = Preprocessing.split_cols(train)

from sklearn.preprocessing import LabelBinarizer
l = LabelBinarizer()
labels = l.fit_transform(authors)
print(labels[:50])

# fix random seed for reproducibility and set max words
num_words = 3

# Preprocess text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(text)
word_sequences = tokenizer.texts_to_sequences(text)
padded_word_sequences = pad_sequences(word_sequences, maxlen=num_words)


# Make model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Embedding
from keras.layers import LSTM

# 19579 rows
# 3 words


model = Sequential()
model.add(Embedding(input_dim=4, output_dim=128, input_length=num_words))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(padded_word_sequences, labels, batch_size=100, epochs=10, verbose=10)

