### Integrate HTML With Flask
### HTTP verb GET And POST
from flask import Flask,redirect,url_for,render_template,request
import math
import cv2
import numpy as np
import time
import random
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
from time import time
import random
import mediapipe as mp
import matplotlib.pyplot as plt


app=Flask(__name__)

# @app.route('/')
# def welcome():
#     return render_template('index_f.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def index_return():
    return render_template('index.html')
# Fitness CHatbot

import nltk
nltk.download('popular')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random

intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app.static_folder = 'static'


@app.route('/game.html')
def game():
    return render_template('game.html')

@app.route('/about.html')
def About():
    return render_template('about.html')

@app.route('/mood_tracker.html')
def mood():
    return render_template('mood_tracker.html')

@app.route('/cycle_tracker.html')
def Game():
    return render_template('cycle_tracker.html')

@app.route('/blogs.html')
def blogs():
    return render_template('blogs.html')

@app.route('/video.html')
def vid():
    return render_template('video.html')

@app.route('/blog-post-1.html')
def blog1():
    return render_template('blog-post-1.html')

@app.route('/video-post-1.html')
def vid1():
    return render_template('video-post-1.html')

@app.route('/blog-post-2.html')
def blog2():
    return render_template('blog-post-2.html')

@app.route('/video-post-2.html')
def vid2():
    return render_template('video-post-3.html')

@app.route('/blog-post-3.html')
def blog3():
    return render_template('blog-post-3.html')
@app.route('/contact_doctor.html')
def contact_doctor():
    return render_template('contact_doctor.html')
@app.route('/video-post-3.html')
def vid3():
    return render_template('video-post-3.html')

@app.route('/blog-post-4.html')
def blog4():
    return render_template('blog-post-4.html')

@app.route('/video-post-4.html')
def vid4():
    return render_template('video-post-4.html')

@app.route('/blog-post-5.html')
def blog5():
    return render_template('blog-post-5.html')

@app.route('/video-post-5.html')
def vid5():
    return render_template('video-post-5.html')

@app.route('/blog-post-6.html')
def blog6():
    return render_template('blog-post-6.html')

@app.route('/video-post-6.html')
def vid6():
    return render_template('video-post-6.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__=='__main__':
    app.run(debug=True)