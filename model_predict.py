import numpy as np
import nltk 
import pickle as pk
import json 
import tensorflow as ts
import random

from data_preprocessing import get_stem_words

ignore_words = ['?', '!',',','.', "'s", "'m"]

model=ts.keras.models.load_model('chatbot_model.h5')

words=pk.load(open('./words.pkl','rb'))

classes=pk.load(open('./classes.pkl','rb'))

intents=json.loads(open('./intents.json').read())

def predict_response(user_input):
    input_token1=nltk.word_tokenize(user_input)
    input_token2=get_stem_words(input_token1,ignore_words)
    input_token3=sorted(list(set(input_token2)))
    bag=[]
    bag_of_words=[]
    for word in words:
        if word in input_token2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    
    bag.append(bag_of_words)

    bag=np.array(bag)
    
    predicted_class_label=model.predict(bag_of_words)

    predicted_class_label=np.argmax(predicted_class_label)

    predicted_class=classes[predicted_class_label]

    for intent in intents['intents']:
        if intent['tag']==predicted_class:
            bot_response=random.choice(intent['responses'])
            return bot_response




print('my name bob,what do u want')

while True:
    user_input=input('gimme ur message')
    print('ur input:',user_input)
    response=predict_response(user_input)
    print("bob's response:",response)
    





