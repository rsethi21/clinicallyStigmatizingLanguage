import argparse
import yaml
import pandas as pd
import os
import pdb

# this script multiprocesses inference at the level of data parallelization

parser = argparse.ArgumentParser()
# input configuration file
parser.add_argument("-i", "--input", help="job configuration file", required=True)

# yaml loading function
def readYaml(fp):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

# breakdown data into n chunks (n in configuration files)
def separate_data(df, n):
    dataframes = []
    for i in range(n):
        if i != n-1:
            dataframes.append(df.iloc[int(len(df.index)/n)*i:int(len(df.index)/n)*(i+1),]) # split into appropriate size
        else:
            dataframes.append(df.iloc[int(len(df.index)/n)*i:,]) # the final split is going to have whatever remains from the division
    return dataframes

# save the dataframe splits for parsing later; saved to a folder path assigned in configurations
def save_data(fp, *dfs):
    filepaths = []
    for i, df in enumerate(dfs):
        filepath = os.path.join(fp, f"{i}.csv")
        filepaths.append(filepath)
        df.to_csv(filepath)
    return filepaths

if __name__ == "__main__":
    args = parser.parse_args()
    # read in configs
    job_configurations = readYaml(args.input)
    # load data
    data = pd.read_csv(job_configurations["multiprocessing_args"]["data_path"], index_col=0)
    # if you want to sample the data or shuffle with a frac of 1.0
    if job_configurations["multiprocessing_args"]["frac"] != None:
        data = data.sample(frac=job_configurations["multiprocessing_args"]["frac"], random_state=job_configurations["multiprocessing_args"]["rs"])
    # load separated dataframes
    dataframes = separate_data(data, job_configurations["multiprocessing_args"]["num_processes"])
    # make output directory for automatic saving
    if not os.path.isdir(job_configurations["multiprocessing_args"]["output_path"]):
        os.system(f"mkdir {job_configurations['multiprocessing_args']['output_path']}")
    # save dataframes in new output dirctory
    filepaths = save_data(job_configurations['multiprocessing_args']['output_path'], *dataframes)
    # iterate dataframes
    for i, fp in enumerate(filepaths):
        # load a separate GPU for parallel processing of data
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{i}"
        # create new directory for outputs of this parallel process
        new_dir_name = f"process_{i}"
        if not os.path.isdir(os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name)):
            os.system(f"mkdir {os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name)}")
        # run script script defined in configurations with the data parallel split; in the configuration you can define what model or approach to use thus can apply the parallel processing to any method
        os.system(
            f"""nohup python3 {job_configurations['multiprocessing_args']['experiment_path']} \
            -p {job_configurations['multiprocessing_args']['model_configs_path']} \
            -d {fp} -o {os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name)} \
            > {os.path.join(os.path.join(job_configurations['multiprocessing_args']['output_path'], new_dir_name), "nohup.out")} 2>&1 &"""
            )