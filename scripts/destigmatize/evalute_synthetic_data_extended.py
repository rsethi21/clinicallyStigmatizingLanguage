import argparse
from tqdm import tqdm
import pandas as pd
import yaml
import pdb
import json

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import BERTScorer
import accelerate

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="csv file of clinical notes with sentence snippets", required=True)
parser.add_argument("-c", "--config", help="yaml file with hyperparameters for LLM and method", required=True)
parser.add_argument("-o", "--output", help="output folder path for newly generated dataset", required=False, default=".")

def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

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

def perplexity(text, model, tokenizer):
    output = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**output, labels=output["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity

def generate(prompt: str, pipeline: object, tokenizer: object, parameters: dict, return_full_text=False):
    generated_outputs = pipeline(
        prompt,
        return_full_text=return_full_text,
        **parameters)
    return [generated_outputs[i]["generated_text"] for i in range(len(generated_outputs))]

def scoring(original, modified, scorer, identity):
    _, __, F1 = scorer.score([original], [modified])
    similarity = F1.mean()
    return similarity

def scoring_perplexity(original, modified, scorer, raw_output, model, tokenizer):
    _, __, F1 = scorer.score([original], [modified])
    similarity = F1.mean()
    liklihood = perplexity(raw_output, model, tokenizer)
    return similarity*(1.0/liklihood)

def process_identity(output):
    prediction = None
    if "yes" in output.lower() and "no" not in output.lower():
        prediction = 1
    else:
        prediction = 0
    return prediction

if __name__ == "__main__":
    
    args = parser.parse_args()
    data = json.load(open(args.input))
    hyperparameters = readYaml(args.config)
    output_path = args.output

    identification_model = AutoModelForCausalLM.from_pretrained(
        hyperparameters["method"]["identification_model"],
        device_map='auto',
        local_files_only=True)
    identification_tokenizer = AutoTokenizer.from_pretrained(hyperparameters["method"]["identification_model"], local_files_only=True)
    identification_pipeline = transformers.pipeline(
        'text-generation',
        model=identification_model,
        tokenizer=identification_tokenizer,
        torch_dtype=torch.float16,
        device_map='auto')

    for i, row in tqdm(enumerate(data), total=len(data)):
        examples = row["examples"]
        examples.insert(0, row["original"])
        predictions = []
        for example in tqdm(examples):
            context = None
            if hyperparameters["method"]["context"] != None:
                context = "\n\n".join(list(pd.read_csv(hyperparameters["method"]["context"])["Context"]))
            identification_prompt = format_it(example, identification_tokenizer, sp = hyperparameters["method"]["identification_prompt"], context = context)
            prediction = generate(identification_prompt, identification_pipeline, identification_tokenizer, hyperparameters["identification_llm"])
            processed_prediction = process_identity(prediction[0])
            predictions.append(processed_prediction)
        data[i]["predictions"] = predictions
        
        
    json.dump(data, open(f"{args.output}/output_sequential.json", "w"), indent=4)