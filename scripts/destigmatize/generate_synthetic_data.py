import argparse
from tqdm import tqdm
import pandas as pd
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import BERTScore
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="csv file of clinical notes with sentence snippets", required=True)
parser.add_argument("-c", "--config", help="yaml file with hyperparameters for LLM and method", required=True)
parser.add_argument("-o", "--output", help="csv output file path for newly generated dataset", required=False, default="preference.csv")

def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def format(inp: str, tokenizer: object, sp: str = None, context: str or None = None):
    messages = []
    if sp != None:
        messages.append({"role": "system", "content": sp})
    if context != None:
        context_priming = f"Please use the context provided below to answer the question.\nContext:\n{context}"
    else:
        context_priming = ""
    messages.append({"role": "user", "content": f"{context_priming}\n\nHere is your query:\n{inp}"})

    custom_template = """{% for message in messages %}
    {% if message['role'] == 'system' %}
    System: {{ message['content'] }}
    {% elif message['role'] == 'user' %}
    User: {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
    Assistant: {{ message['content'] }}
    {% endif %}
    {% endfor %}
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
        **parameters["llm"])
    return [generated_outputs[i]["generated_text"] for i in len(generated_outputs)]

def scoring(original, modified, scorer, identity):
    _, __, F1 = scorer.score([original], [modified])
    similarity = F1.mean()
    return similarity*identity

def scoring_perplexity(original, modified, scorer, output_identity, model, tokenizer):
    _, __, F1 = scorer.score([original], [modified])
    similarity = F1.mean()
    perplexity = evaluate.load("perplexity")
    probability = perplexity.compute(predictions=[output_identity], model=model, tokenizer=tokenizer)
    return similarity*probability[0]

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

    scoring_model = None
    identification_model = None
    identification_tokenizer = None
    identification_pipeline = None
    modifying_model = None
    modifying_tokenizer = None
    modifying_pipeline = None

    for i, row in data.iterrows():
        text = row[hyperparameters["method"]["column"]]
