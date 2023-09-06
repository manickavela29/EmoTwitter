from transformers import pipeline
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification

import urllib.request
import csv

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def load_model(type:str = 'quant') :
    model_path = './model/'+type
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #The models are optimized for CPU inference, therefore no support for GPU execution is provided as of now
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path)
    inference = pipeline("text-classification", model=ort_model, tokenizer=tokenizer)

    return inference




# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]