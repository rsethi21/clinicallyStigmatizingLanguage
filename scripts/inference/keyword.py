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

if __name__ == "__main__":
    # arguments
    args = parser.parse_args()

    # parameters
    parameters = readYaml(args.configurations_fp)
    
    # openning csv file
    input_df = pd.read_csv(args.notes_fp)
    input_notes = list(input_df["Assessment"])

    outputs = []
    for i in tqdm(input_notes, desc="Notes"):
        i = i.lower()
        i = i.split(" ")
        o = 0
        for check in parameters["stigmatizing_terms"]:
            if check in i:
                o = 1
            else:
                pass
        outputs.append(o)
    
    with open(os.path.join(args.output_fp, "generated_outputs.json"), "w") as outfile_generation:
        json.dump(outputs, outfile_generation)