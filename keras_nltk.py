# nltk module
import nltk
from nltk.stem.lancaster import LancasterStemmer

# module we need for Tensorflow
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json

# create a stemmer
stemmer = LancasterStemmer()

# load intents
# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


words = []
classes = []
documents = []
ignore_words = []


for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word into a sentence
        w = nltk.word_tokenize(pattern)
        # add word to our words list
        words.extend(w)
        # add documents to our corpus
        documents.append((w, intent['tag']))

        # add to our classses
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# stem and lower case the word and remove duplicates
words =  [stemmer.stem(w.lower()) for w in words ]




# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique stemmed words", words)
