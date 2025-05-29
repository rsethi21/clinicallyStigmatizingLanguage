import os
import yaml
import json
import argparse
from tqdm import tqdm
import pdb

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np

# arguments for commandline interface
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--configurations_fp", help="path to configurations file for experiment", required=True)
parser.add_argument("-d", "--notes_fp", help="path to file with clinical notes in csv format", required=True)
parser.add_argument("-o", "--output_fp", help="path to folder for outputs", required=True)

def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def format_message(sp: str, input: str):
    """
    Purpose: create chat dialog following appropriate format.
    Input: system prompt, context, input
    Output: formatted prompt
    """
    start = "<|begin_of_text|>\n" # start token
    system_prompt = f"<|start_header_id|>system<|end_header_id|>{sp}\n<|eot_id|>\n" # question/task
    input_data = f"<|start_header_id|>user<|end_header_id|>\nBelow is the clinical note to use for the question:\n{input}<|eot_id|>\n" # user input
    end = "<|start_header_id|>assistant<|end_header_id|>" # end token
    return start+system_prompt+input_data+end

def generate(prompt: str, pipeline: object, tokenizer: object, parameters: dict):
    # generating output
    generated_outputs = pipeline(
        prompt,
        **parameters["llm"])
    # parsing llama3 output
    start_index = generated_outputs[0]['generated_text'].rindex('<|start_header_id|>assistant<|end_header_id|>') + len("<|start_header_id|>assistant<|end_header_id|>")
    generated_text = generated_outputs[0]['generated_text'][start_index:]
    generated_text = generated_text.replace("\n", "")
    generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
    return generated_text

if __name__ == "__main__":
    # arguments
    args = parser.parse_args()

    # parameters
    parameters = readYaml(args.configurations_fp)

    # instantiating model and pipeline
    tokenizer = AutoTokenizer.from_pretrained(parameters["method"]["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        parameters["method"]["model_path"],
        device_map='auto',
        local_files_only=True)
    pipeline_tg = transformers.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map='auto')
    
    # openning csv file
    input_df = pd.read_csv(args.notes_fp)
    input_notes = list(input_df["Assessment"])

    # iterate notes and infer
    outputs = []
    for i in tqdm(input_notes, desc="Notes"):
        prompt = format_message(parameters["method"]["sp"], i)
        outputs.append(generate(prompt, pipeline_tg, tokenizer, parameters))
    
    with open(os.path.join(args.output_fp, "generated_outputs.json"), "w") as outfile_generation:
        json.dump(outputs, outfile_generation)