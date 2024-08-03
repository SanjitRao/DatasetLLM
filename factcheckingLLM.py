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


import llama7bchat

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

### WESBSITE SCRAPER (implements general_text_parser_newspaper3k)
def newsApi_text_scraper_newspaper3k(topic):
    topic_articles = newsApi.get_everything(q=topic)['articles']
    topic_urls = [x['url'] for x in topic_articles]
    topic_texts = [general_text_parser_newspaper3k(url) for url in topic_urls]
    
    topic_docs = [Document(topic_urls[i], text_summarizer(topic_texts[i]), topic_texts[i]) for i in range(len(topic_texts))]

    return topic_docs

summarizer = pipeline("summarization")
def text_summarizer(text): # want to take article text and summarize it
    rv = summarizer(text, min_length=5, max_length=100)
    print(rv)
    if (len(rv) !=0):
        return rv[0]["summary_text"]
    return "NA"

        
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


## LLAMA MODEL PIPELINE
from huggingface_hub import login
login(token="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")

from transformers import AutoTokenizer
import transformers
import torch
import json 
import re

model_id = "meta-llama/Llama-2-7b-chat-hf"

"""tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipeline.save_pretrained("pipeline_llama27Bchat")
tokenizer.save_pretrained("tokenizer_llama27Bchat") """

from transformers import pipeline, AutoTokenizer, LlamaForCausalLM

# Replace with your custom model of choice
model = LlamaForCausalLM.from_pretrained('pipeline_llama27Bchat')
tokenizer = AutoTokenizer.from_pretrained('tokenizer_llama27Bchat')

pipe = pipeline(task='text-generation',  # replace with whatever task you have
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map='auto')


## try asking llama if sentence and summarized text agree
def agreement_detected(sentence, text):
    rv = False
    summarized_text = text_summarizer(text)
    if summarized_text == "NA":
        return rv

    #remove any \n from sentence
    sentence = sentence.replace("\n", "")

     # Final Prompt Formatting
    prompt = "Do the sentence '" + sentence + "' and the text '" + text + "' agree? Return a reponse in JSON format." 
    prompt +="\n"


    # make sure prompt is w/o \n (prompt = prompt.replace("\n", ""))
    sequences = pipe(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        #eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    # Extract Response
    response = sequences[0]["generated_text"]

    # Remove \n's from response
    response = response.replace("\n", "")
    print(response)

    return rv

## testing
sentence = "Dogs are the best."
text = "There have been many studies that have tried to determine which animal is superior. After months of experiements, dogs rank among the best."

agreement = agreement_detected(sentence, text)
print(agreement)
