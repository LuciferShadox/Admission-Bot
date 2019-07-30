from flask import Flask, render_template, request
import os
import nltk
import warnings
warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages
import re
import numpy as np
import random
import string # to process standard python strings

#L1 = Label(window, text="Input")
#E1 = Entry(window, bd =10)


f=open('C:/Users/Rahul Ramakrishnan/my_project/dataset.txt','r',errors = 'ignore')

raw=f.read()
raw=raw.lower()# converts to lowercase
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)#
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]



# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        
    else:
        robo_response = robo_response+sent_tokens[idx]
    try:
        robo_response=robo_response.splitlines()[1]
    except:
        #nothing
        print()
    sent_tokens.remove(user_response)    
    return robo_response

def evaluate(user_response):   
    #user_response=resp.get() 
    #resp.delete(0, END) 
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            
            text="You are welcome.."
        else:
            if(greeting(user_response)!=None):
                text=greeting(user_response)
            else:
                text=response(user_response)                
    else:
        
        text=" Bye! take care.."
    return text





app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')
@app.route('/process',methods=['POST'])
def process():
	user_input=request.form['user_input']

	bot_response=evaluate(user_input)
	bot_response=str(bot_response)
	#print("Friend: "+bot_response)
	return render_template('index.html',user_input=user_input,bot_response=bot_response)

if __name__=='__main__':
	app.run(debug=True,port=5002)