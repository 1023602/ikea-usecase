#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from textblob import TextBlob
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from flask import Flask, render_template, request
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from transformers import pipeline
import logging



NAIVE_BAYES = 'mnb_dict.pkl'
stop_list = stopwords.words('english')


clas = pipeline("zero-shot-classification")
labels = ['Minor', 'Major', 'Critical']


def clean_text(email_text):
    '''
    This fucntion takes the string as input and removes punctuations,
    numberics, stopwords and extra spaces.
    '''
    email_text = [c if c not in string.punctuation else " " for c in email_text]
    email_text = ''.join(email_text).lower()
    email_text = re.sub("xxx*","", email_text)
    email_text = email_text.split()
    email_text = [word for word in email_text if word not in stop_list if word.isalpha() if len(word) > 1]
    return ' '.join(email_text)


def predict_product_mnb(mnb_file, email_text):
    '''
    This function takes file name that contain Naviebayes classifier and text as input
    and returns the poduct name
    '''
    # Load the ML model pickle file
    with open(mnb_file, 'rb') as f:
        model = pickle.load(f)
    
    cv = model["cv"]
    mnb = model["mnb"]
    
    # Cleant the text data
    email_text = clean_text(email_text)
    # run the prediction using ML model
    cv_out = cv.transform(pd.Series(email_text))
    product = mnb.predict(cv_out)
    
    return product[0]


def predict_severity(email_text):
    '''
    This function predicts the severity of complaint using zero-shot-classification
    from transformers
    '''
    out = clas(email_text, labels)
    return out['labels'][0]


def explain_model(mnb_file, email_text, product):
    '''
    This function return the explnation for ML model predictions
    using LIME
    '''

    with open(mnb_file, 'rb') as f:
        model = pickle.load(f)
    
    cv = model["cv"]
    mnb = model["mnb"]
    
    map_dict = {}
    for index, val in enumerate(mnb.classes_):
        map_dict[val] = index
    
    c = make_pipeline(cv, mnb)
    email_text = clean_text(email_text)
   
    explainer = LimeTextExplainer(class_names = list(mnb.classes_))
    exp = explainer.explain_instance(email_text, 
                                     c.predict_proba,
                                     num_features = 10, 
                                     labels=(map_dict[product],))
    
    return exp
    


app = Flask(__name__)


@app.route("/")     
def basic_home():
    return render_template('index.html')  


@app.route("/", methods=['POST'])
def result():
    if request.method == 'POST':
        email_text = request.form.get("complaint")
        
        # Predicting model
        product = predict_product_mnb(NAIVE_BAYES, email_text)
        
        #Predicting severity
        email_severity = predict_severity(email_text)
        
        # Model Explanation
        exp = explain_model(NAIVE_BAYES, email_text, product)
        exp = exp.as_html()
        
        return render_template('model_predictions.html', result =product, severity= email_severity, exp = exp)


if __name__ == "__main__":
    app.run(debug=True)
     
        
        
        
        