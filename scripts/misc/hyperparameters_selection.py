import argparse
import yaml
import pandas as pd
import numpy as np
import os
import subprocess
from itertools import product
import pdb
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="job configuration file", required=True)

def readYaml(fp):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def writeYaml(fp, dictionary):
    with open(fp, "w") as file:
        yaml.dump(dictionary, file, default_flow_style=False)

def save_data(fp, *dfs):
    filepaths = []
    for i, df in enumerate(dfs):
        filepath = os.path.join(fp, f"{i}.csv")
        filepaths.append(filepath)
        df.to_csv(filepath)
    return filepaths

def parameter_sets(parameter_dict):
    combos = list(product(*list(parameter_dict.values())))
    parameter_keys = list(list(parameter_dict.keys()))
    parameter_sets = []
    for combo in combos:
        combo_set = {key: value for key, value in zip(parameter_keys, combo)}
        parameter_sets.append(combo_set)
    return parameter_sets

def check_util(t):
    all_util = []
    for _ in range(t):
        time.sleep(1)
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
        gpu_utilization = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
        all_util.append(gpu_utilization)
    return np.mean(np.array(all_util))

if __name__ == "__main__":
    args = parser.parse_args()
    job_configurations = readYaml(args.input)
    combinations = parameter_sets(job_configurations["parameters"])
    for i, combo in tqdm(enumerate(combinations), total=len(combinations), desc="Parameter Set"):
        output_path = os.path.join(job_configurations['output_path'], f'output_{i}')
        if not os.path.isdir(output_path):
            os.system(f"mkdir {output_path}")
        method_configs = {"llm": job_configurations["llm"], "method": combo}
        writeYaml(os.path.join(output_path, f"configs_set_{i}.yml"), method_configs)
        multiprocessing_dictionary = job_configurations["multiprocessing_args"]
        multiprocessing_dictionary["model_configs_path"] = os.path.join(output_path, f"configs_set_{i}.yml")
        multiprocessing_dictionary["output_path"] = output_path
        multiprocess_configs = {"multiprocessing_args": job_configurations["multiprocessing_args"]}
        writeYaml(os.path.join(output_path, f"multiprocess_configs_set_{i}.yml"), multiprocess_configs)
        os.system(f"python3 {job_configurations['script']} -i {os.path.join(output_path, f'multiprocess_configs_set_{i}.yml')}")
        gpu_utilization = check_util(20)
        print()
        while gpu_utilization != 0:
            gpu_utilization = check_util(20)
            print(f"Running parameter set #{i+1}", end='\r')