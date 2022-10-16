import random
import json
import pickle
import numpy as np
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam  # type: ignore


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data.json').read())

words = []
classes = []
documents = []
training = []


def clean_words(Text):
    # Replacing all non-alphabetic characters with a space
    sms = re.sub('[^a-zA-Z]', ' ', Text)
    sms = sms.lower()  # converting to lowecase
    sms = sms.split()
    sms = ' '.join(sms)
    return sms


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return [word for word in text if word not in stop_words]


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(clean_words(sentence))
    sentence_words = remove_stopwords(sentence_words)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = clean_up_sentence(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('word.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


def bag_of_words(sentence_words):
    bag = [0]*len(words)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1
    return bag


for document in documents:
    bag = bag_of_words(document[0])
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
X_train = list(training[:, 0])
y_train = list(training[:, 1])

print(len(max(X_train)), len(min(X_train)))

model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dense(len(y_train[0]), activation='softmax'))

adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

hist = model.fit(np.array(X_train), np.array(y_train),
                 epochs=150, batch_size=5, verbose=2)  # type: ignore

# model.save('model.h5', hist)  # type: ignore
