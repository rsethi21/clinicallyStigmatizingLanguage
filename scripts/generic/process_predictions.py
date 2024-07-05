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
parser.add_argument("-p", "--configurations_fp", help="path to configurations file for RAG experiment", required=True)
parser.add_argument("-d", "--notes_fp", help="path to file with clinical notes in csv format", required=True)
parser.add_argument("-o", "--output_fp", help="path to folder for outputs", required=True)

def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def format_message(sp: str, input: str, model_v: str = "llama3"):
    """
    Purpose: create chat dialog following appropriate format.
    Input: system prompt, context, input, model version
    Output: formatted prompt
    """
    # asserting only certain formatting for chat prompt
    assert model_v in ["llama2", "llama3", "none"], "Only formatting allowed: llama2, llama3, none (which is llama2)."
    # formatting llama2/mistral/mixtral prompt
    if model_v == "llama2" or model_v == "none":
        start = "[INST]\n" # start token
        system_prompt = f"<<SYS>>\n{sp}\n<</SYS>>\n" # question/task
        input_data = f"Below is the critique you generated:\n{input}\n" # user input
        end = "[/INST]\n\n" # end token
    # formatting llama3 prompt
    elif model_v == "llama3":
        start = "<|begin_of_text|>\n" # start token
        system_prompt = f"<|start_header_id|>system<|end_header_id|>{sp}\n<|eot_id|>\n" # question/task
        input_data = f"<|start_header_id|>user<|end_header_id|>\nBelow is the critique you generated:\n{input}\n" # user input
        end = "<|start_header_id|>assistant<|end_header_id|>" # end token
    return start+system_prompt+input_data+end

def generate(prompt: str, pipeline: object, tokenizer: object, parameters: dict):
    # asserting only certain formatting for output parsing purposes
    assert parameters["method"]["formatting"] in ["llama2", "llama3", "none"], "Only formatting allowed: llama2, llama3, none (which is llama2)."
    # generating output
    generated_outputs = pipeline(
        prompt,
        **parameters["llm"])
    # parsing llama3 output
    if parameters["method"]["formatting"] == "llama3":
        start_index = generated_outputs[0]['generated_text'].rindex('<|start_header_id|>assistant<|end_header_id|>') + len("<|start_header_id|>assistant<|end_header_id|>")
        generated_text = generated_outputs[0]['generated_text'][start_index:]
        generated_text = generated_text.replace("\n", "")
        generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
    # parsing llama2, mistral/mixtral output
    else:
        end_index = generated_outputs[0]['generated_text'].index('[/INST]')
        generated_text = generated_outputs[0]['generated_text'][end_index+len("[/INST]"):]
        generated_text = generated_text.replace("\n", "")
        generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
    return generated_text.lower()

if __name__ == "__main__":
    # arguments
    args = parser.parse_args()

    # parameters
    parameters = readYaml(args.configurations_fp)

    # instantiating model and pipeline
    if parameters["method"]["pipeline"]:
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
    input_notes = list(pd.read_csv(args.notes_fp)["generated_outputs.json"])
    try:
        for index, inp in enumerate(input_notes):
            input_notes[index] = eval(inp)
    except:
        pass
    if parameters["method"]["pipeline"]:
        outputs = []
        # iterate notes and apply RAG
        for i in tqdm(input_notes, desc="Notes"):
            if type(i) == str:
                prompt = format_message(parameters["method"]["sp"], i, model_v=parameters["method"]["formatting"])
                outputs.append(generate(prompt, pipeline_tg, tokenizer, parameters))
            elif type(i) == list:
                final_predictions = []
                for item in i:
                    prompt = format_message(parameters["method"]["sp"], item, model_v=parameters["method"]["formatting"])
                    final_predictions.append(generate(prompt, pipeline_tg, tokenizer, parameters))
                outputs.append(",".join(final_predictions))
    else:
        outputs = input_notes
    if type(outputs[0]) == list:
        outputs = [",".join(output) for output in outputs]
    processed_auto = [1 if "yes" in str(entry).lower() else 0 for entry in outputs]

    with open(os.path.join(args.output_fp, "predictions.json"), "w") as outfile_generation:
        json.dump(processed_auto, outfile_generation)