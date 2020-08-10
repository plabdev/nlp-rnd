
# nltk module
import nltk
from nltk.stem.lancaster import LancasterStemmer

# module we need for Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json


# fb magic bot
#ACCESS_TOKEN = 'EAAESHHaUabIBADcrZAhdigkIPkjZBNZAIrDEunBGC69Jpwv7trAjZBewYum0zl2X1GjTeRMAGsciA6oj9SKZAI3yvDj4rtaVxfcmZBgdXbwAvicHepeLfq9FWfznUZC3sSgIwnXLheaxu3zKEdxJePPDy75e4Vj6x76HZAzX8bcYqZCybfq7ob5L5'   #ACCESS_TOKEN = os.environ['ACCESS_TOKEN']

# fb bot
ACCESS_TOKEN = 'EAAKrLyeQf6oBAEsp3gBfZAk3PPxDufVnziHk4RtYK6RSfHgasKxKC8l9Aj0iTlhH0oPLxcOjQ7e9nu7E8rQZC5mZCMtpA5YVo0vcx1kwqUhvGbWcPO53EPYfsCcLL3BZCZBikuQojLGIlYuqnI9Ey5vX91oTBbJCFL3rRuSW2KPZC46I7eL0Q6'   #ACCESS_TOKEN = os.environ['ACCESS_TOKEN']

VERIFY_TOKEN = 'JAHIDTOEKN@'   #VERIFY_TOKEN = os.environ['VERIFY_TOKEN']




# create a stemmer
stemmer = LancasterStemmer()

# load intents
# import our chat-bot intents file
import json
with open('intents_robi_en.json') as json_data:
    intents = json.load(json_data)


#Use pickle to load  data
data = pickle.load( open( "./pickle/robi_en_data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

#Use pickle to load in the pre-trained model
with open('./pickle/robi_en_model.pkl', 'rb') as f:
    global model
    model = pickle.load(f)

confidence_level = 0.5

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


def classify(sentence):
    # Add below two lines for workaround error _thread._local' object has no attribute 'value'
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    global model
    global confidence_level
    ERROR_THRESHOLD = confidence_level
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])

    results = model.predict([input_data],workers=1,verbose=0)[0]

    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    return return_list


def botResponse(sentence):
    import random
    tagResult = classify(sentence)
    if not tagResult:
        return "Sorry, I do not understand"
    responseList = list(filter(lambda x:x['tag'] == tagResult[0][0],intents['intents']))
    return random.choice(responseList[0]['responses'])

if __name__ == "__main__":
    texts='hi'
    auto_reply = botResponse(texts)
    print(auto_reply)
