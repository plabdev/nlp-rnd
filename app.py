#Python libraries that we need to import for our bot
import random
from flask import Flask, request
from pymessenger.bot import Bot
from credential import *
import os
import webview
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2



#Image classifier
# *************************************************************************************************
from skimage import io

LABEL = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
model = load_model("/home/alpha/Desktop/PreNeu/Office Day/13 Aug/nlp-rnd/keras weights/mnist_cnn.h5")
model.summary()
load_input = model.input
input_shape= list(load_input.shape)

height = int(input_shape[1])
width = int(input_shape[2])
channel = int(input_shape[3])
print(height, width, channel)

def load_image(image):

    if channel == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    image = cv2.resize(image,(height,width)) # resize image to 32x32
    image = image.reshape(1, height, width,channel).astype('float32')
    image = np.array(image) / 255

    return image

def image_classification(url):
    image = io.imread(url)
    image = load_image(image)

    result = model.predict(image)
    result = np.argmax(result, axis=1)

    text = "Yes! It is  "+str(LABEL[result[0]])

    return text



# **********************************************************************************************

app = Flask(__name__)
bot = Bot (ACCESS_TOKEN)

#We will receive messages that Facebook sends our bot at this endpoint
sample_responses = ["Thanks"]
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook."""
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
       output = request.get_json()
       for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
            if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                recipient_id = message['sender']['id']

                #print(message)
                if message['message'].get('text'):
                    text_message = message['message']['text']
                    response_sent_text = get_texts(texts=text_message)
                    send_message(recipient_id, response_sent_text)
                #if user sends us a GIF, photo,video, or any other non-text item
                # elif message['message'].get('attachments'):
                #     attachments_message = message['message']['attachments']
                #     response_sent_nontext = get_attachments(attachments = attachments_message)
                #     send_message(recipient_id, response_sent_nontext)

    return "Message Processed"


def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token you sent
    #if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        print("varified")
        return request.args.get("hub.challenge")
    # else:
    #     return 'Invalid verification token'



def get_texts(texts):

    texts = texts.lower()
    print(texts)
    #global sample_responses

    auto_reply = botResponse(texts)

    # return selected item to the user
    return auto_reply


def get_attachments(attachments):

    attachments = attachments[0]
    print(attachments['payload']['url'])
    url = attachments['payload']['url']

    # print(attachments['type'], type(attachments['type']))
    #
    # if attachments['type'] == 'image':
    #     result = image_classification(url)
    #     return result
    # else:

    return "unknow attachments"


#uses PyMessenger to send response to user
def send_message(recipient_id, response):
    #sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

if __name__ == "__main__":
    app.run()
