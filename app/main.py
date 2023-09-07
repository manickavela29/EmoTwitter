# Importing Necessary modules
from transformers import pipeline
from transformers import AutoTokenizer,RobertaTokenizer,TextClassificationPipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.special import softmax
from typing import List
import onnxruntime as ort
import numpy as np
from time import perf_counter
import uvicorn
import urllib
import csv

labels = ['anger', 'joy', 'optimism', 'sadness']

def preprocess(tweets):
    ptweets = []
    for tweet in tweets :
        new_text = []
        for t in tweet.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        new_text = " ".join(new_text)
        ptweets.append(new_text)
    return ptweets

#inference function is running on modle optimized for CPU, GPU support is not available
def inference(tweetslist,type='quant') :
    model_path = 'twitter-models/'+type
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    ort_session = ort.InferenceSession(model_path+'/model.onnx',providers=['DnnlExecutionProvider'])
    
    model_path = 'EmoTwitter/app/twitter-models/base/'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #The models are optimized for CPU inference, therefore no support for GPU execution is provided as of now
    ort_session = ort.InferenceSession(model_path+'/model.onnx',providers=['DnnlExecutionProvider'])

    for tweet in tweetslist :
        inputs = tokenizer(tweet, add_special_tokens=True,return_tensors="np")
        inputs = {k: [vi.astype(np.int64) for vi in v] for k,v in inputs.items()} # handling in windows data type size
        outputs_name = ort_session.get_outputs()[0].name
        outputs = ort_session.run(output_names=[outputs_name], input_feed=inputs)
        print(outputs)

        scores = outputs[0][0]

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        scores = softmax(scores)
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")

# Creating FastAPI instance
app = FastAPI()

class Tweets(BaseModel):
    texts : List[str]

# Defining path operation for root endpoint
@app.get('/')
def index():
    return {'message': 'Welcome to Twitter Emotion Prediction'}

# Defining path operation for /name endpoint
@app.post('/tweet-base')
def twitter_base_detect(tweet : Tweets):
    # Defining a function that takes only string as input and output the
    # following message.
    
    tweet = preprocess(tweet.texts)
    inference = load_pipe('base')
    start_time = perf_counter()
    pred = inference(tweet)
    latency = perf_counter() - start_time
    return {'message' : {"quant_model":False,"emotion-detection" : pred,"latency" : str(latency)+' ms'}}

# Defining path operation for /name endpoint
@app.post('/tweet-quant')
def twitter_quant_detect(tweet:Tweets):
    # Defining a function that takes only string as input and output the
    # following message.
    tweet = preprocess(tweet.texts)
    inference = load_pipe('quant')
    start_time = perf_counter()
    pred = inference(tweet)
    latency = perf_counter() - start_time
    return {'message' : {"quant_model":True,"emotion-detection" : pred,"latency" : str(latency)+' ms'}}

if __name__ == "__main__" :
    uvicorn.run(app,host="0.0.0.0",port=8000)