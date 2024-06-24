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

def generate_critique(prompt: str, pipeline: object, tokenizer: object, parameters: dict):
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
    return generated_text

def generate_revision(prompt: str, pipeline: object, tokenizer: object, parameters: dict):
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
        try:
            start = list(generated_text).index("{")
            end = list(generated_text).index("}")
        except:
            start = -1
            end = len(generated_text)
    # parsing llama2, mistral/mixtral output
    else:
        end_index = generated_outputs[0]['generated_text'].index('[/INST]')
        generated_text = generated_outputs[0]['generated_text'][end_index+len("[/INST]"):]
        generated_text = generated_text.replace("\n", "")
        generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
        try:
            start = list(generated_text).index("{")
            end = list(generated_text).index("}")
        except:
            start = -1
            end = len(generated_text)
    return generated_text[start+1:end]

def process_multiprocessed_output(combined_list):
    flattened_list = []
    for i in range(len(combined_list)):
        flattened_list.extend(combined_list[i])
    pairs = []
    for revisions in flattened_list:
        for j in range(len(revisions)-1):
            pairs.append([revisions[j], revisions[j+1]])
    return flattened_list, pairs

def sft_formatting(pairs, sp):
    data = []
    for pair in pairs:
        entry = {
            "instruction": sp,
            "input": pair[0],
            "output": pair[1]
        }
        data.append(entry)
    return data

def dpo_formatting(pairs, sp):
    data = []
    for pair, s in zip(pairs, sp):
        entry = {
            "instruction": s,
            "chosen": pair[1],
            "rejected": pair[0]
        }
        data.append(entry)
    return data

def format_message(sp: str, input: str, context: str or None = None, model_v: str = "llama3"):
    """
    Purpose: create chat dialog following appropriate format.
    Input: system prompt, context, input, model version
    Output: formatted prompt
    """
    # asserting only certain formatting for chat prompt
    assert model_v in ["llama2", "llama3", "none"], "Only formatting allowed: llama2, llama3, none (which is llama2)."
    # RAG context string
    if context != None:
        context_priming = f"Please use the critique provided below to answer the question.\nCritique:\n{context}"
    else:
        context_priming = ""
    # formatting llama2/mistral/mixtral prompt
    if model_v == "llama2" or model_v == "none":
        start = "[INST]\n" # start token
        system_prompt = f"<<SYS>>\n{sp}\n<</SYS>>\n" # question/task
        input_data = f"Below is the clinical note to use for the question:\n{input}\n" + context_priming # user input
        end = "[/INST]\n\n" # end token
    # formatting llama3 prompt
    elif model_v == "llama3":
        start = "<|begin_of_text|>\n" # start token
        system_prompt = f"<|start_header_id|>system<|end_header_id|>{sp}\n<|eot_id|>\n" # question/task
        input_data = f"<|start_header_id|>user<|end_header_id|>\nBelow is the clinical note to use for the question:\n{input}\n{context_priming}<|eot_id|>\n" # user input
        end = "<|start_header_id|>assistant<|end_header_id|>" # end token
    return start+system_prompt+input_data+end


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
    input_notes = list(input_df["IdentifiedSentence"])
    constitution = pd.read_csv(parameters["method"]["sp"])
    constitution

    all_critiques = []
    all_revisions = []
    all_dpo_prompts = []
    # samples = parameters["method"]["samples"]
    # iterate notes
    for _ in tqdm(range(parameters["method"]["epochs"]), desc="Epoch", total=parameters["method"]["epochs"]):
        critiques = []
        revisions = [] # use all revision, input pairs for training purposes
        for pos, i in tqdm(enumerate(input_notes), desc="Notes", total=len(input_notes)):
            if not parameters["method"]["vary"]:
                bylaw = constitution.sample(n=1) # should I force different one for each epoch or just allow random (look into difference)
            temp_critiques = []
            temp_revisions = [i]
            temp_input = i
            for __ in tqdm(range(parameters["method"]["attempts"]), desc="Attempts", total=parameters["method"]["attempts"]):
                if parameters["method"]["vary"]:
                    bylaw = constitution.sample(n=1)
                critique = None
                prompt = format_message(bylaw.critique_request.values[0], temp_input, context=critique, model_v=parameters["method"]["formatting"])
                temp_critiques.append(generate_critique(prompt, pipeline_tg, tokenizer, parameters))
                critique = temp_critiques[__]
                prompt = format_message(bylaw.revision_request.values[0], temp_input, context=critique, model_v=parameters["method"]["formatting"])
                temp_revisions.append(generate_revision(prompt, pipeline_tg, tokenizer, parameters))
                temp_input = temp_revisions[__]
                all_dpo_prompts.append(bylaw.alignment_request.values[0])
            critiques.append(temp_critiques)
            revisions.append(temp_revisions)
        all_critiques.append(critiques)
        all_revisions.append(revisions)
    with open(os.path.join(args.output_fp, "critiques.json"), "w") as critique_file:
        json.dump(all_critiques, critique_file)
    with open(os.path.join(args.output_fp, "revisions.json"), "w") as revision_file:
        json.dump(all_revisions, revision_file)
    with open(os.path.join(args.output_fp, "alignment_prompt.json"), "w") as prompt_file:
        json.dump(all_dpo_prompts, prompt_file)