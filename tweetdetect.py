from flask import Flask, request , make_response,render_template

from tweet_utils import preprocess,load_model
import numpy as np
import os,csv

app = Flask(__name__)

'''
@app.route("/",methods=["GET","POST"])
def index():
    errors = []
    results = {}
    return render_template('index.html', errors=errors, results=results)
    # return f"Twitter Emotion detector optimized for CPU inference"
'''
@app.route("/",methods=["GET","POST"])
@app.route("/tweet-emotion-quant", methods=["GET","POST"])
def quant():
    errors = []
    results = {}

    if request.method == "POST" :
        tweet = request.form['url']

        cpu_inference_pipe = load_model()

        #text = request.args.get('tweet')
        text = preprocess(tweet)

        preds = cpu_inference_pipe(text)
        results['label'] = preds[0]['label']
    return render_template('index.html', errors=errors, results=results)
    #return '<h1> Emotion for the tweet : {}</h1>'.format(preds[0]['label'])

@app.route("/tweet-emotion-base", methods=["GET","POST"])
def base():
    errors = []
    results = {}

    if request.method == "POST" :
        
        tweet = request.form['url']

        cpu_inference_pipe = load_model('base')

        #text = request.args.get('tweet')
        text = preprocess(tweet)

        preds = cpu_inference_pipe(text)
        results['label'] = preds[0]['label']
    return render_template('index.html', errors=errors, results=results)
    #return '<h1> Emotion for the tweet : {}</h1>'.format(preds[0]['label'])

#if __name__ == '__main__' :


