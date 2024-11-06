### access openAI using LangChain, create small dataset of textual data using 
### newspaper API, ask for pandas DF format (columns, num_rows, etc), and set up 
### vectorization algo to find top 10 sources and parse info
from newsapi import NewsApiClient
import pandas as pd
import numpy as np
from urllib.request import urlopen, Request
#from bs4 import BeautifulSoup
import requests
import re
from textblob import TextBlob
from newspaper import Article
from transformers import pipeline
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer


import llama31chat

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'}
api_key = "af1752b5cbfe44fcb3e41e452ca10881"
newsApi = NewsApiClient(api_key=api_key)
topics_list = ["Facebook", "Tesla", "Apple"]

# --- CREATING THE DATABASE --- #
class Document:
    link = ''
    summary = ''
    text = ''
    def __init__(self, l, s, t):
      self.link = l
      self.summary = s
      self.text = t

    def get_link(self):
      return self.link

    def get_text(self):
      return self.text

    def get_summary(self):
      return self.summary

### GENERAL TEXT PARSER (NEWSPAPER3K)
def general_text_parser_newspaper3k(url): # takes article URL and outputs article text
    print(url)
    
    try:
      article = Article(url)
      article.download()
      article.parse()
      text = article.text
      return text
    
    except Exception as e:
      return "NA"

summarizer = pipeline("summarization")
def text_summarizer(text): # want to take article text and summarize it
    rv = summarizer(text, min_length=5, max_length=100)
    print(rv)
    if (len(rv) !=0):
        return rv[0]["summary_text"]
    return "NA"

### WESBSITE SCRAPER (implements general_text_parser_newspaper3k)
def newsApi_text_scraper_newspaper3k(topic):
    topic_articles = newsApi.get_everything(q=topic)['articles']
    topic_urls = [x['url'] for x in topic_articles]
    topic_texts = [general_text_parser_newspaper3k(url) for url in topic_urls]
    
    topic_docs = [Document(topic_urls[i], text_summarizer(topic_texts[i]), topic_texts[i]) for i in range(len(topic_texts))]

    return topic_docs

# --- CREATING DATASET TEMPLATE --- #
from torch.utils.data import Dataset

class DatasetTemplate(Dataset):
    def __init__ (self, **kwargs): # {length : ???, num_colums : ???, column_names : ???}
        try:
            self.length = kwargs['length']
            self.num_colms = kwargs['num_columns']
            self.column_names = kwargs['column_names']
            self.data = [] 
        except Exception as e:
            print("Error: malformed or missing input")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pass

    def fill_data(self, curated_data): # curated_data = list(stuff)
        for data in curated_data:
            self.data.append(data)
        
# ---  QUESTION-ANSWER GENERATION PIPELINE  --- #

### QUESTION_GENERATION MODEL
model_name = "allenai/t5-small-squad2-question-generation"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
def run_QG_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output

### QUESTION_ANSWERING MODEL
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
def run_QA_model(question, context):

    result = question_answerer(question=question, context=context)
    return result['answer'], result['score']

def question_answer_generator(sentence): # takes a sentence and generates a question-answer-context pairing
    question = run_QG_model(sentence)
    answer = run_QA_model(question, sentence)
    
    return [question, answer, sentence] # [question, answer, context] format

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
def question_answer_dataframe_generator(document): # takes a document and produces question-answer pairings
    data = []
    # TODO split text into sentences along '.' and "\n"
    data_sentences = sent_tokenize(document.text)
    for sentence in data_sentences:
        data.append(question_answer_generator(sentence))
    
    df = pd.DataFrame(data, columns=["Question", "Answer", "Context"])
    
    return df

def data_generator(document, task, column_names): # takes a document, task, column_names and extracts data using Llama27bchat
    rv = {name:[] for name in column_names}
    sentences = sent_tokenize(document.text)

    for sentence in sentences:
        is_useful = llama31chat.filter_useful_data(task, sentence)
        if is_useful:
            data = llama31chat.extract_data(sentence, column_names)

            for name in column_names:
                for key in data.keys():
                    if name in key:
                        rv[name].append(data[key]) # In theory, this should work...


    df = pd.DataFrame(rv)



    return df

### TESTING
import pickle

with open('database.pickle', 'rb') as handle:
    database = pickle.load(handle)

#print(database[0].text)
document = database[0]
prompt = "Data about Facebook ads"
column_names = ['sentiment', 'question', 'answer']

data = data_generator(document, prompt, column_names)
print(data.head())

with open('facebook_ad_data', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('facebook_ad_data.pickle', 'rb') as handle:
    ad_data = pickle.load(handle)

print(ad_data, len(ad_data))
