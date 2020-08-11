# -*- coding: utf-8 -*-
"""Text_Analytics_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rKHjS5evQ7R8dHZnBfM7dBKIfwp8q0xk
"""

#Import libraries needed
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
nltk.download('stopwords')
nltk.download('punkt')

#Load econbiz dataset (use only a portion to see pre-processing part)
econ_df = pd.read_csv("econbiz.csv")
#econ_df_use = econ_df[econ_df["fold"].isin(range(0,10))]
econ_df_use = econ_df[['id','title','labels']]

#Separate labels (currently separeted by tab space) and this is needed for the multilabelbinarizer
econ_df_use['labels'] = econ_df_use['labels'].str.split()

#Pre-process title data
def preprocess_title(title):

    stop_words = set(stopwords.words('english')) #Stop words to remove
    title_processed = re.sub('[^a-zA-Z]', ' ', title) #Removes numbers
    title_processed = re.sub(r"\s+[a-zA-Z]\s+", ' ', title_processed) # Removes single characters
    title_processed = re.sub(r'\s+', ' ', title_processed) # Removes multiple spaces
    title_tokens = word_tokenize(title_processed) #Tokenize the title sample for checking stopwords
    title_processed = [token for token in title_tokens if not token in stop_words]

    return title_processed

#Preprocess title column
X = []
title_unprocessed = list(econ_df_use['title'])

for idx,title in enumerate(title_unprocessed):
  X.append(preprocess_title(title))

#Preprocess labels
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(econ_df_use.labels)
labels = multilabel_binarizer.classes_
y = multilabel_binarizer.transform(econ_df_use.labels)

#Divide dataset into training and validation set (80/20)
#To this point, the title column still needs to be vectorize
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9000)

print("Data summary")
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#Initialize tokenizer from keras that will vectorize title values
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

vocabulary_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(x_train, padding= 'post',maxlen=51)
x_test = pad_sequences(x_test,padding='post',maxlen=51)
print('Pad sequences (samples x time)')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

#Got the model structure and parameters from https://keras.io/examples/imdb_bidirectional_lstm/ which is a BiLSTM example for text classification

model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=51))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=32,
          epochs=2,
          validation_data=[x_test, y_test])

probs = np.arange(0.05,1.0,0.05)
scores = []
for prob in probs:
  preds = model.predict(x_test)
  preds[preds>=prob] = 1
  preds[preds<prob] = 0
  scores.append(tuple((f1_score(y_test, preds, average="samples"),prob)))
  print(tuple((f1_score(y_test, preds, average="samples"),prob)))

