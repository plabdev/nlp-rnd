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


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
