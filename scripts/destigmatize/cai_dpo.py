# imports non-AI
import yaml
import json
import pandas as pd
import argparse
from tqdm import tqdm
import pdb

# imports AI-specific
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import accelerate

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="csv file of clinical notes with sentence snippets", required=True)
parser.add_argument("-c", "--config", help="yaml file with hyperparameters for LLM and method", required=True)
parser.add_argument("-o", "--output", help="output folder to save csv and model", required=False, default=".")

# read yaml
def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

# formatting prompt function
def format_it(inp: str, tokenizer, sp: str = None, context: str or None = None):
    messages = []
    if sp != None:
        messages.append({"role": "system", "content": sp})
    if context != None:
        context_priming = f"Please use the context provided below to answer the question.\nContext:\n{context}"
    else:
        context_priming = ""
    messages.append({"role": "user", "content": f"{context_priming}\n\nHere is your query:\n{inp}"})

    custom_template = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% endif %}
    <|eot_id|>
    {% for message in messages[1:] %}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>
    {{ message['content'] }}<|eot_id|>
    {% endfor %}
    {% if add_generation_prompt %}
    <|start_header_id|>assistant<|end_header_id|>
    {% endif %}
    """
    
    try:
        input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    except:
        tokenizer.chat_template = custom_template
        input_text=tokenizer.apply_chat_template(messages, tokenize=False)

    return input_text

# generation function
def generate(prompt: str, pipeline: object, tokenizer: object, parameters: dict, return_full_text=False):
    generated_outputs = pipeline(
        prompt,
        return_full_text=return_full_text,
        **parameters)
    return [generated_outputs[i]["generated_text"] for i in range(len(generated_outputs))]

# running on command line
if __name__ == "__main__":
    pass

    ## instantiate arguments, data, hyperparameters
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    hyperparameters = readYaml(args.config)

    ## load models
    critique_model = None
    critique_tokenizer = None
    revision_model = None
    revision_tokenizer = None

    ## data loop
    ## provide current clinical note (probably those considered stigmatizing) as "initial response" and request critique using the constitution
        ### during inference will need to chunk the clinical note to locate the area where the langauge can be found

    ## provide critique to develop revision
        ### randomly draw principles from the constitution

    ## repeat top two steps n times

    ## use initial and final response for SFT

    ## provide revisions for DPO

    ## save model and data