import gradio as gr
import argparse
import yaml
import pandas as pd
import json

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #, BitsAndBytesConfig
from bert_score import BERTScorer
import accelerate

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="yaml file with hyperparameters for LLM and method", required=True)
args = parser.parse_args()

def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def format_it(inp: str, tokenizer, sp: str = None, context: str or None = None):
    messages = []
    if sp != None:
        messages.append({"role": "system", "content": sp})
    if context != None:
        context_priming = f"Please use the chat history provided below to answer the question:\n{context}"
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

def process_identity(output):
    prediction = None
    if "yes" in output.lower() and "no" not in output.lower():
        prediction = 1
    else:
        prediction = 0
    return prediction

configs = readYaml(args.config)
# c = BitsAndBytesConfig(load_in_4bit=True)
c = None
model_ident = AutoModelForCausalLM.from_pretrained(configs["method"]["identification_model"], local_files_only=True, device_map="auto")
tokenizer_ident = AutoTokenizer.from_pretrained(configs["method"]["identification_model"], local_files_only=True)
pipeline_id = pipeline("text-generation", model = model_ident, tokenizer = tokenizer_ident)
model_md = AutoModelForCausalLM.from_pretrained(configs["method"]["modification_model"], local_files_only=True, quantization_config = c, device_map="auto")
tokenizer_md = AutoTokenizer.from_pretrained(configs["method"]["modification_model"], local_files_only=True)
pipeline_md = pipeline("text-generation", model = model_md, tokenizer = tokenizer_md)
scoring_model = BERTScorer(model_type=configs["method"]["bert"])
fnc = lambda x: "yes" if x == 1 else "no"

def chat_bot(instruction, other, model, query, history):
    history = history or []
    context = ""
    for n, entry in enumerate(history):
        context = context + f"\n\nOutput from run #{n+1}" + "\n" + entry[0] + "\n" + entry[1] + "\n-------------"
    if len(context) == 0:
        context = None
    hps = configs["identification_llm"]
    tokenizer = tokenizer_md
    pipeline = pipeline_md
    sp = None
    if model == 1:
        pipeline = pipeline_md
        tokenizer = tokenizer_md
        hps = configs["modifying_llm"]
    elif model == 2:
        pipeline = pipeline_id
        tokenizer = tokenizer_ident
    if instruction == 1:
        sp = configs["method"]["identification_prompt"]
        context = None
    if instruction == 2:
        sp = configs["method"]["modifying_prompt"]
    if instruction == 3:
        sp = "You are a physician, and you have identified that the following clinical note contains stigmatizing language in a substance use context. Please report the entire sentence or sentences you found to be stigmatizing in a substance use context and explain why you believe so."
        pipeline = pipeline_id
        tokenizer = tokenizer_ident
        context = None
    if instruction == 4:
        sp = other
        pipeline = pipeline_md
        tokenizer = tokenizer_md

    final_query = query
    prompt = format_it(final_query, tokenizer, sp = sp, context = context)
    answer = generate(prompt, pipeline, tokenizer, hps)
    if instruction == 2:
        try:
            index = example.index("{")
            answer = json.loads(answer[0][index:])[configs["method"]["json_header"]]
        except:
            pass
        identification_prompt = format_it(answer, tokenizer_ident, sp = configs["method"]["identification_prompt"], context = context)
        prediction = generate(identification_prompt, pipeline_id, tokenizer_ident, configs["identification_llm"])
        processed_prediction = process_identity(prediction[0])
        final_score = scoring(query, answer, scoring_model, 1-int(processed_prediction))
        history.append([final_query, f"Unprocessed output: {answer[0]}\nThe note is stigmatizing?: {fnc(processed_prediction)}\nRatio of similarity to original: {final_score}"])
        return history, history
    elif instruction == 1:
        processed_prediction = process_identity(answer[0])
        history.append([final_query, f"The note is stigmatizing?: {fnc(processed_prediction)}"])
        return history, history
    else:
        history.append([final_query, f"Unprocessed output: {answer[0]}"])
        return history, history


demo = gr.Interface(chat_bot, [gr.Dropdown(["", "identify", "modify", "clarify", "custom"], type="index"), "text", gr.Dropdown(["", "pre-trained", "fine-tuned"], type="index"), "text", "state"], ["chatbot", "state"])
demo.launch()