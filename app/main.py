# Importing Necessary modules
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from time import perf_counter
from transformers import pipeline
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


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

def load_pipe(type:str = 'quant') :
    model_path = './twitter-models/'+type
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #The models are optimized for CPU inference, therefore no support for GPU execution is provided as of now
    print(model_path)
    onnx_model = ORTModelForSequenceClassification.from_pretrained(model_path)
    infpipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
    return infpipe

# Creating FastAPI instance
app = FastAPI()

class Tweets(BaseModel):
    texts : List[str]

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Twitter Emotion Prediction'}

# Defining path operation for /name endpoint
@app.post('/tweet-base/')
def twitter_base_detect(tweet : Tweets):
    # Defining a function that takes only string as input and output the
    # following message.

    tweet = preprocess(tweet.texts)
    inference = load_pipe('base')
    start_time = perf_counter()
    pred = inference(tweet)
    latency = perf_counter() - start_time
    pred = inference(tweet)
    return {'message' : {"quant_model":False,"emotion-detection" : pred,"latency" : str(latency)+' ms'}}

# Defining path operation for /name endpoint
@app.post('/tweet-quant/')
def twitter_quant_detect(tweet:Tweets):
    # Defining a function that takes only string as input and output the
    # following message.
    tweet = preprocess(tweet.texts)
    inference = load_pipe('quant')
    start_time = perf_counter()
    pred = inference(tweet)
    latency = perf_counter() - start_time
    pred = inference(tweet)
    return {'message' : {"quant_model":True,"emotion-detection" : pred,"latency" : str(latency)+' ms'}}