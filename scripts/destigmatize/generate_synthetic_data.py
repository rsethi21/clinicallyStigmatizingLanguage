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
parser.add_argument("-o", "--output", help="csv output file path for newly generated dataset", required=False, default="preference.csv")

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
    data = pd.read_csv(args.input)
    hyperparameters = readYaml(args.config)
    output_path = args.output

    scoring_model = BERTScorer(model_type=hyperparameters["method"]["bert"])
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
    modifying_model = AutoModelForCausalLM.from_pretrained(
        hyperparameters["method"]["modification_model"],
        device_map='auto',
        local_files_only=True)
    modifying_tokenizer = AutoTokenizer.from_pretrained(hyperparameters["method"]["modification_model"], local_files_only=True)
    modifying_pipeline = transformers.pipeline(
        'text-generation',
        model=modifying_model,
        tokenizer=modifying_tokenizer,
        torch_dtype=torch.float16,
        device_map='auto')
    
    if hyperparameters["method"]["frac"] != None:
        sampled_data = data.sample(frac=hyperparameters["method"]["frac"])
        sampled_data.reset_index(inplace=True, drop=True)
    else:
        sampled_data = data

    out_data = []
    for i, row in tqdm(sampled_data.iterrows(), total=len(sampled_data.index), desc="Data Entry..."):
        text = row[hyperparameters["method"]["column"]]
        prompt = format_it(text, modifying_tokenizer, sp = hyperparameters["method"]["modifying_prompt"], context = None)
        examples = generate(prompt, modifying_pipeline, modifying_tokenizer, hyperparameters["modifying_llm"])
        examples.insert(0, row[hyperparameters["method"]["column"]])
        scores = []
        predictions = []
        fail = []
        for n, example in enumerate(examples):
            try:
                index = example.index("{")
                example = json.loads(example[index:])[hyperparameters["method"]["json_header"]]
                examples[n] = example
            except:
                fail.append(n)
            context = None
            if hyperparameters["method"]["context"] != None:
                context = "\n\n".join(list(pd.read_csv(hyperparameters["method"]["context"])["Context"]))
            identification_prompt = format_it(example, identification_tokenizer, sp = hyperparameters["method"]["identification_prompt"], context = context)
            prediction = generate(identification_prompt, identification_pipeline, identification_tokenizer, hyperparameters["identification_llm"])
            processed_prediction = process_identity(prediction[0])
            final_score = scoring(text, example, scoring_model, 1-int(processed_prediction))
            # final_score = scoring_perplexity(text, example, scoring_model, f"{identification_prompt}\n<|start_header_id|>assistant<|end_header_id|>{prediction}<|eot_id|>", identification_model, identification_tokenizer)
            scores.append(final_score.item())
            predictions.append(processed_prediction)
        out_data.append({"original": text, "examples": examples, "scores": scores, "predictions": predictions, "fail": fail})
    json.dump(out_data, open(f"{args.output}/output_review.json", "w"), indent=4)