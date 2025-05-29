import json
import pandas as pd
import os
import argparse
import pdb

# this script combines the outputs of multiprocessing to one final prediction array

parser = argparse.ArgumentParser()
# arguments
parser.add_argument("-i", "--input", help="folder path of outputs to merge", required=True) # folder with all the outputs to merge (this is from the multiprocessing scripts)
parser.add_argument("-o", "--output", help="file path to store csv output", required=True) # final output file to store all the predictions in

if __name__ == "__main__":
    args = parser.parse_args()
    # read in all the folders created during multiprocessing
    process_folders = [os.path.join(args.input, process_folder) for process_folder in os.listdir(args.input) if "csv" not in process_folder and "log" not in process_folder and "json" not in process_folder and "txt" not in process_folder and "yml" not in process_folder]
    process_folders = sorted(process_folders)
    # store log files in a separate directory
    logs = os.path.join(args.input, "log_files")
    try:
        os.system(f"mkdir {logs}")
    except:
        pass
    for i, f in enumerate(process_folders):
        os.system(f"mv {os.path.join(f, 'nohup.out')} {os.path.join(logs, f'nohup_{i}.out')}")
    # store file names with predictions (all subprocesses have the same file names from the multiprocess scripts)
    data_to_merge = {file: [] for file in os.listdir(process_folders[0])}
    # iterate each subprocess file and append predictions
    for file in sorted(list(data_to_merge.keys())):
        for folder in process_folders:
            if os.path.isdir(folder):
                data_to_merge[file].extend(json.load(open(os.path.join(folder, file), "r")))
    # return output as csv
    if "csv" in args.output:
        df = pd.DataFrame.from_dict(data_to_merge)
        df.to_csv(args.output)
    # return output as json
    elif "json" in args.output:
        json.dump(data_to_merge, open(args.output, "w"))
    # remove accessory folders
    try:
        os.system(f"rm -r {os.path.join(args.input, 'process*')}")
    except:
        pass