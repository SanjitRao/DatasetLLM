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

def dict_format(column_names): # return dictionary format for column names
    rv = '{'

    for i in range(len(column_names)):
        if i == len(column_names) - 1:
            rv += '"' + column_names[i] + '"' + ': ...'
        else:
            rv += '"' + column_names[i] + '"' + ': ..., '

    rv += '}'
    return rv


import json 
def extract_data(sentence, column_names): #takes sentence, column_names and finds data in dictionary format
    #remove any \n from sentence
    sentence = sentence.replace("\n", "")

    names = ""
    # concatenate column_names in string format (assumes names are in string type)
    for i in range(len(column_names)-1):
        names += column_names[i] + ", "
    names += column_names[len(column_names)-1] + " "
    # names = "a, b, c " (space at the end)

    # Final Prompt Formatting
    prompt = "Find " + names + "for the sentence '" + sentence + "' and return the result in a Python dictionary with the keys in double quotes." 
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
    

    # Extract Llama's returned string dictonary (return substring within {})
    pattern_find_dict = r"\{([^}]*)\}"
    response = re.findall(pattern_find_dict, response)
    print(response)

    # Convert response string into dictionary
    if (len(response) !=0): # have to make sure that list aint empty

        response = response[0]
        print(response) 
        



        response = response.replace("'", '"')
        response = '{' + response + '}'
        try:
            data_dict = json.loads(response)
            return data_dict
        
        except Exception as e:
            return {name: 'NA' for name in column_names}



        '''# replace all quotes with empty space
        response = response.replace('"', '')
        print(response)
        response = response.replace("'", " ")
        print(response)

        data_dict = json.loads(response)
        return data_dict'''



        # TODO use regex to re.compile(pattern w/ :), m.span that shit
        '''pattern_find_keys = re.compile("(\"\w+\"[:]+)")
        list_spans = []
        for m in pattern_find_keys.finditer(response):
            print(m.span(), m.group())
            list_spans.append(m.span())

        data_dict = {}
        # run through list_spans to find keys and coresponding values
        for i in range(len(list_spans)-1): # does NOT give back the last key-valu pair
            key = response[list_spans[i][0]: list_spans[i][1]] # includes the :

            # Remove the colon
            key = key[:-1]

            # Find value associated with key
            value = response[list_spans[i][1] + 1: list_spans[i+1][0]]
            data_dict[key] = value
        
        # Find the last key-value pair and store it in the dictionary
        last_key = response[list_spans[len(list_spans)-1][0]: list_spans[len(list_spans)-1][1]]
        last_key = last_key[:-1]
        last_value = response[list_spans[len(list_spans)-1][1] + 1: len(response)]

        data_dict[last_key] = last_value'''

        #return data_dict
    
    return {name: 'NA' for name in column_names} ## TODO in document-level data extraction, making sure we check for  bool(empty_dict) == False

def filter_useful_data(task, sentence):
    # Ask Llama27B model whether or not the given sentence is relavent to task
    # parse out [YES] or [NO] using regex
    # return True or False
    # returns False by default
    
    #remove any \n from sentence
    sentence = sentence.replace("\n", "") 

    prompt = f"Is the sentence '{sentence}' relevant to answering the question '{task}'? Answer with '[YES]' if yes or '[NO]' if no."

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
    print(response)

    # Remove \n's from response
    response = response.replace("\n", "")

    # Find the [YES] or [NO]
    pattern = r"\[\w+\]"
    response = re.findall(pattern, response)
    print(response)
    print()

    if len(response) !=0:
        response = response[-1]
        print(response, type(response))
        # remove []
        response = response.replace("[", "")
        response = response.replace("]", "")

        if response == "YES":
            return True

    return False



### TESTING

test_sentence = "Meta is gearing up to debut a new type of Facebook ads that will allow users in the European Union to download apps without having to visit their mobile platform's app store, according to The Verge."
column_names = ['sentiment', 'question', 'answer']
task = "Information about Facebook ads."
rv = extract_data(test_sentence, column_names)

#rv = filter_useful_data(task, test_sentence)
print(rv)


'''TODO
1) Document-level extraction using extract_data() and filter_useful_data()
2) Better prompt engineering for extract() and filter()
3) Finding most relevant Documents for task (preferablly efficiently)
4) Construct website UI
5) Construct github repo w/ good documentation


6) Webcrawling for useful data
7) Fast loading of Llama27B model 
8) Connect model to Internet

'''

                                        