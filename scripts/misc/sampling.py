#!/usr/bin/env python3

import transformers, torch, os, numpy, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object
from time import time
import pdb
from tqdm import tqdm
import os
import json
import concurrent.futures as cf
from itertools import repeat
import numpy as np
import argparse
import data
import re
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="file with separated data", required=True)
parser.add_argument("-p", "--parameters", help="file with model configurations", required=True)
parser.add_argument("-o", "--output", help="folder path for output", required=True)

def readYaml(fp):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def generate(prompt_text, pipeline, tokenizer, parameters):
  generated_outputs = pipeline(
      prompt_text,
      eos_token_id=tokenizer.eos_token_id,
      **parameters)
  # remove the the prompt from output and evaluate
  end_index = generated_outputs[0]['generated_text'].index('[/INST]')
  generated_text = generated_outputs[0]['generated_text'][end_index+7:]
  generated_text = generated_text.replace("\n", "")
  generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
  return generated_text

def main(parameters, outdir, inputs):
  """Ask for input and feed into llama2"""
  # https://medium.com/@pankaj_pandey/guidelines-for-prompting-large-language-models-b598189abed5
  # the source above is an article that provides information about how to word a prompt
  tokenizer = AutoTokenizer.from_pretrained(parameters["method"]["model_path"], local_files_only=True)
  model = AutoModelForCausalLM.from_pretrained(
    parameters["method"]["model_path"],
    device_map='auto',
    load_in_4bit=True, local_files_only=True)
  pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map='auto')

  generated_outputs = []
  for io in tqdm(inputs):
    temp_out = generate(io, pipeline, tokenizer, parameters["llama"])
    generated_outputs.append(temp_out)

  with open(os.path.join(outdir, "generated_text.json"), "w") as generated:
    json.dump(generated_outputs, generated)
  os.system(f"chmod go-rwx {outdir}/*")

if __name__ == "__main__":

  args = parser.parse_args()
  inputs = data.probing_inputs(args.data)
  main(readYaml(args.parameters), args.output, inputs)