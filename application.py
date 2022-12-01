# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:56:53 2019

@author: charl
"""

import flask
from flask import json

import boto3
import fasttext as ft
#import fastText as ft
#import urllib.request

import sys
import os
import datetime

# Cleaning libraries
from bs4 import BeautifulSoup
import re
import itertools
import emoji

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


# Tweet lexicon sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


##############################################################################################
#
#   INIT
#
##############################################################################################

analyser = SentimentIntensityAnalyzer()
model_path = '/tmp/'

SENTIMENT_THRESHOLD = 0.40


def download_from_s3(s3, bucket, keyname, filelocation):
    print("Download: " + keyname)
    s3.download_file(bucket, keyname, filelocation)

# GET YOUR MODELDS FROM A S3 BUCKET
#s3 = boto3.client('s3')
#download_from_s3(s3, 'YOUR-S3-BUCKET',"model-en.ftz",'/tmp/model-en.ftz')
#download_from_s3(s3, 'YOUR-S3-BUCKET',"model-es.ftz",'/tmp/model-es.ftz')
#download_from_s3(s3, 'YOUR-S3-BUCKET',"model-fr.ftz",'/tmp/model-fr.ftz')
#download_from_s3(s3, 'YOUR-S3-BUCKET',"model-it.ftz",'/tmp/model-it.ftz')
#download_from_s3(s3, 'YOUR-S3-BUCKET',"model-de.ftz",'/tmp/model-de.ftz')


##############################################################################################
#
#   CLEANING FUNCTIONS
#
##############################################################################################


def strip_accents(text):
    # length_initial=len(text)
    # initial_text=text
    if 'ø' in text or 'Ø' in text:
        return text
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def load_dict_smileys():

    return {
        ":‑)": "smiley",
        ":-]": "smiley",
        ":-3": "smiley",
        ":->": "smiley",
        "8-)": "smiley",
        ":-}": "smiley",
        ":)": "smiley",
        ":]": "smiley",
        ":3": "smiley",
        ":>": "smiley",
        "8)": "smiley",
        ":}": "smiley",
        ":o)": "smiley",
        ":c)": "smiley",
        ":^)": "smiley",
        "=]": "smiley",
        "=)": "smiley",
        ":-))": "smiley",
        ":‑D": "smiley",
        "8‑D": "smiley",
        "x‑D": "smiley",
        "X‑D": "smiley",
        ":D": "smiley",
        "8D": "smiley",
        "xD": "smiley",
        "XD": "smiley",
        ":‑(": "sad",
        ":‑c": "sad",
        ":‑<": "sad",
        ":‑[": "sad",
        ":(": "sad",
        ":c": "sad",
        ":<": "sad",
        ":[": "sad",
        ":-||": "sad",
        ">:[": "sad",
        ":{": "sad",
        ":@": "sad",
        ">:(": "sad",
        ":'‑(": "sad",
        ":'(": "sad",
        ":‑P": "playful",
        "X‑P": "playful",
        "x‑p": "playful",
        ":‑p": "playful",
        ":‑Þ": "playful",
        ":‑þ": "playful",
        ":‑b": "playful",
        ":P": "playful",
        "XP": "playful",
        "xp": "playful",
        ":p": "playful",
        ":Þ": "playful",
        ":þ": "playful",
        ":b": "playful",
        "<3": "love"
    }


def load_dict_contractions():

    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "e'er": "ever",
        "em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "he've": "he have",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I'm'a": "I am about to",
        "I'm'o": "I am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "I've": "I have",
        "kinda": "kind of",
        "let's": "let us",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "ne'er": "never",
        "o'": "of",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "shalln't": "shall not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "'tis": "it is",
        "'twas": "it was",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've": "would have",
        "y'all": "you all",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "Whatcha": "What are you",
        "luv": "love"
    }


def tweet_cleaning_for_sentiment_analysis(tweet):
    # Escaping HTML characters
    tweet = BeautifulSoup(tweet).get_text()
    tweet = tweet.replace('\x92', "'")

    # REMOVAL of hastags/account
    tweet = ' '.join(
        re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
    # Removal of address
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
    # Removal of Punctuation
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
    # LOWER CASE
    tweet = tweet.lower()

    # Apostrophe Lookup #https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    APPOSTOPHES = load_dict_contractions()
    tweet = tweet.replace("’", "'")
    words = tweet.split()
    reformed = [APPOSTOPHES[word]
                if word in APPOSTOPHES else word for word in words]
    tweet = " ".join(reformed)
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

    # Deal with EMOTICONS
    # https://en.wikipedia.org/wiki/List_of_emoticons
    # {"<3" : "love", ":-)" : "smiley", "" : "he is"}
    SMILEY = load_dict_smileys()
    words = tweet.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)
    tweet = emoji.demojize(tweet)

    # Strip accents
    tweet = strip_accents(tweet)
    tweet = tweet.replace(":", " ")
    tweet = ' '.join(tweet.split())
    return tweet


##############################################################################################
#
#   INFERENCE SYSTEM
#
##############################################################################################

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model_en = None                # Where we keep the model when it's loaded
    model_es = None
    model_fr = None
    model_it = None
    model_de = None

    @classmethod
    def get_models(cls, language):
        model_selected = None

        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model_en == None:  # and language.lower() =="en":
            cls.model_en = ft.load_model(
                os.path.join(model_path, 'model-en.ftz'))
        if cls.model_es == None and language.lower() == "es":
            cls.model_es = ft.load_model(
                os.path.join(model_path, 'model-es.ftz'))
        if cls.model_fr == None and language.lower() == "fr":
            cls.model_fr = ft.load_model(
                os.path.join(model_path, 'model-fr.ftz'))
        if cls.model_it == None and language.lower() == "it":
            cls.model_it = ft.load_model(
                os.path.join(model_path, 'model-it.ftz'))
        if cls.model_de == None and language.lower() == "de":
            # ft.load_model('/tmp/model-de.ftz')
            cls.model_de = ft.load_model(
                os.path.join(model_path, 'model-de.ftz'))

        """Loading it if it's not already loaded."""
        if language.lower() == "en":
            model_selected = cls.model_en
        elif language.lower() == "es":
            model_selected = cls.model_es
        elif language.lower() == "fr":
            model_selected = cls.model_fr
        elif language.lower() == "it":
            model_selected = cls.model_it
        elif language.lower() == "de":
            model_selected = cls.model_de
        else:
            model_selected = cls.model_en

        return model_selected

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
           Dictionnary with text to analyze and language in which it is written."""
        clf = cls.get_models(input['language'])

        return clf.predict(input['text'], k=3)


##############################################################################################
#
#   FLASK APP
#
##############################################################################################
# The flask app for serving predictions
application = flask.Flask(__name__)


@application.route('/', methods=['GET'])
def onion():
    #### modal loading ####
    loaded_modal = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_modal = pickle.load(fid)

    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        loaded_modal = pickle.load(vd)

    ##############################
    # how to use model to predic
    prediction = loaded_modal.predict(
        vectorizer.transform(['This is fake news']))[0]
    return (prediction)
    # output will be 'FAKE' if fake, 'REAL' if real


@application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_models(
        "") is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@application.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data.
    """
    data = None

    if flask.request.content_type == 'application/json':
        #data = flask.request.data.decode('utf-8')
        data = flask.request.get_json()

    else:
        return flask.Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')

    text = tweet_cleaning_for_sentiment_analysis(data.get('text'))
    tweet = {'text': text,
             'language': data.get('language')}

    # Do the prediction
    predictions = ScoringService.predict(tweet)
    # print(predictions)
    t = {
        'prob': list(predictions[1]),
        'label': list(predictions[0])
    }
    score = {}
    if len(t['label']) > 0 and t['label'][0] != "__label__":
        score[t['label'][0].replace('__label__', '').title()] = t['prob'][0]
    if len(t['label']) >= 2 and t['label'][1] != "__label__":
        score[t['label'][1].replace('__label__', '').title()] = t['prob'][1]
    if len(t['label']) >= 3 and t['label'][2] != "__label__":
        score[t['label'][2].replace('__label__', '').title()] = t['prob'][2]

    if not 'MIXED'.title() in score:
        score['MIXED'.title()] = 0
    if not 'POSITIVE'.title() in score:
        score['POSITIVE'.title()] = 0
    if not 'NEGATIVE'.title() in score:
        score['NEGATIVE'.title()] = 0
    if not 'NEUTRAL'.title() in score:
        score['NEUTRAL'.title()] = 0

    final_sentiment = t['label'][0].replace('__label__', '').upper()

    # REDUCE NEUTRAL BY UPRANKING NEXT SENTIMENT IF ABOVE A THRESHOLD
    if final_sentiment == "NEUTRAL":
        if len(t['label']) >= 2 and len(t['prob']) >= 2:  # check if exist
            if t['label'][1].upper() in ['__LABEL__NEGATIVE', '__LABEL__POSITIVE'] and t['prob'][1] > SENTIMENT_THRESHOLD:
                final_sentiment = t['label'][1].replace(
                    '__label__', '').upper()

    # PREPARING OUTPUT
    if final_sentiment == "":
        final_sentiment = 'NEUTRAL'.upper()

    # VADER SENTIMENT ANALYSIS IF SENTIMENT IS NEUTRAL  #positive sentiment: compound score >= 0.05
    try:
        if final_sentiment == "NEUTRAL":
            vader = analyser.polarity_scores(data.get('text'))
            score_temp = {}
            if vader["compound"] > 0.05 and vader["neu"] < 0.50:
                final_sentiment = "POSITIVE"
                score_temp['POSITIVE'.title()] = vader["pos"] * -1
                score_temp['NEGATIVE'.title()] = vader["neg"] * -1
                score_temp['NEUTRAL'.title()] = vader["neu"] * -1
                score_temp['MIXED'.title()] = 0
                score = score_temp
            elif vader["compound"] < -0.05 and vader["neu"] < 0.50:
                final_sentiment = "NEGATIVE"
                score_temp['POSITIVE'.title()] = vader["pos"] * -1
                score_temp['NEGATIVE'.title()] = vader["neg"] * -1
                score_temp['NEUTRAL'.title()] = vader["neu"] * -1
                score_temp['MIXED'.title()] = 0
                score = score_temp
    except:
        print("Error Vader")

    result = {'Sentiment': final_sentiment,
              'SentimentScore': score
              }

    return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')


if __name__ == '__main__':
    application.run()
