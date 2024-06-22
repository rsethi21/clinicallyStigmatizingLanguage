import argparse
import yaml
import pandas as pd
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="job configuration file", required=True)

def readYaml(fp):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

def separate_data(df, n):
    dataframes = []
    for i in range(n):
        if i != n-1:
            dataframes.append(df.iloc[int(len(df.index)/n)*i:int(len(df.index)/n)*(i+1),])
        else:
            dataframes.append(df.iloc[int(len(df.index)/n)*i:,])
    return dataframes

def save_data(fp, *dfs):
    filepaths = []
    for i, df in enumerate(dfs):
        filepath = os.path.join(fp, f"{i}.csv")
        filepaths.append(filepath)
        df.to_csv(filepath)
    return filepaths

if __name__ == "__main__":
    args = parser.parse_args()
    job_configurations = readYaml(args.input)
    data = pd.read_csv(job_configurations["multiprocessing_args"]["data_path"], index_col=0)
    if job_configurations["multiprocessing_args"]["frac"] != None:
        data = data.sample(frac=job_configurations["multiprocessing_args"]["frac"], random_state=job_configurations["multiprocessing_args"]["rs"])
    dataframes = separate_data(data, job_configurations["multiprocessing_args"]["num_processes"])
    if not os.path.isdir(job_configurations["multiprocessing_args"]["output_path"]):
        os.system(f"mkdir {job_configurations['multiprocessing_args']['output_path']}")
    filepaths = save_data(job_configurations['multiprocessing_args']['output_path'], *dataframes)
    for i, fp in enumerate(filepaths):
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{i}"
        new_dir_name = f"process_{i}"
        if not os.path.isdir(os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name)):
            os.system(f"mkdir {os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name)}")
        os.system(
            f"""nohup python3 {job_configurations['multiprocessing_args']['experiment_path']} \
            -p {job_configurations['multiprocessing_args']['model_configs_path']} \
            -d {fp} -o {os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name)} \
            > {os.path.join(os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name), "nohup.out")} 2>&1 &"""
            )