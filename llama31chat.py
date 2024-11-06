from huggingface_hub import login
login(token="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")

import transformers
import torch
import json 
import re
import pandas as pd
import numpy as np
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

# --- PROOF OF CONCEPT ----------------------------------------------

'''json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}



prompt = "Generate a person's information based on the following schema:"
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)'''

# ----------------------------------------------------------------

def generate_objects(num_rows, column_names):
    data_objects = {}

    for i in range(num_rows):
        data_objects["row_{}".format(i)] = []
    
        for name in column_names:
            data_objects["row_{}".format(i)].append({"name" : name[0], 
                                                     "type" : name[1]})


    return data_objects


# Test generate_ojects()
'''num_rows = 10
column_names = [
    ["cars", "string"],
    ["batman", "integer"],
    ["isHim", "boolean"]
]

data_objs = generate_objects(num_rows, column_names)
print(data_objs)'''

def generate_json_schema_for_object(properties):
    schema = {
        "type": "object",
        "properties": {}
    }

    for prop in properties:
        if 'name' in prop and 'type' in prop:
            schema['properties'][prop['name']] = {
                "type": prop['type']
            }

            # Handle nested objects or arrays
            if prop['type'] == "object" and 'properties' in prop:
                schema['properties'][prop['name']]['properties'] = prop['properties']
            elif prop['type'] == "array" and 'items' in prop:
                schema['properties'][prop['name']]['items'] = prop['items']
    
    return schema

def generate_json_schema_for_multiple_objects(data_objects):
    schema = {
        "type": "object",
        "properties": {}
    }

    for object_name, properties in data_objects.items():
        schema['properties'][object_name] = generate_json_schema_for_object(properties)
    
    return schema


def generate_data(num_rows, column_names, article_text):
    '''
    num_rows: number of datapoints to exist in the dataset: each object in the output JSON will be named "row_{}".format(i)
    column_names: [ [name (str), type (str)] ]
    article_text: str, will be passed in from direct user input and integrated into a standard prompt
    '''
    # idea 1: generate a JSON with each object being a row in the dataset: each row_i has a list of properties that correspond to column names
    # prompt = "Given the bracketed summary of the article [{}], generate a dataset using the following schema:".format(article_text)
    
    data_objects = generate_objects(num_rows, column_names)

    schema = generate_json_schema_for_multiple_objects(data_objects)
    prompt = "Given the bracketed summary of the article [{}], generate a dataset using the following schema:".format(article_text)
    jsonformer = Jsonformer(model, tokenizer, schema, prompt)
    generated_data = jsonformer()

    return generated_data


## TEST generate_data:
num_rows = 10
column_names = [
    ["cars", "string"],
    ["batman", "string"],
    ["hasCar", "string"]
]

article_text = "There are 5 batmen in the world, who all have cars. The first batman has a Tesla, the second batman has a Mustang, the third has a lamborghini, the fourth has a Tesla, and the last batman has a BMW"

generated_data = generate_data(num_rows, column_names, article_text)
print(generated_data)













                                        