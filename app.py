import pickle
import numpy as np
import re
import json
import random

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from tensorflow.python.keras.models import load_model
from flask import Flask, request

intents = json.loads(open('data.json').read())
model = load_model('model.h5')
words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# nltk.download('popular')
lemmatizer = WordNetLemmatizer()


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


def bag_of_words(sentence_words):
    bag = [0]*len(words)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1
    return bag


def predict_class(sentence):
    bow = bag_of_words(clean_up_sentence(sentence))
    res = model.predict(np.array([np.array(bow)]))[0]  # type: ignore
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probabilty': str(r[1])})

    return return_list


def get_reposnse(intents_list, intent_json):
    tag = intents_list[0]['intent']
    list_of_intents = intent_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/message/get")
def get_bot_response():
    message = request.args.get('msg')  # type: ignore
    print(message)
    ints = predict_class(message)
    return get_reposnse(ints, intents)


if __name__ == "__main__":
    app.run()
