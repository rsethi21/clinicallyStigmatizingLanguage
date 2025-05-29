from sklearn.metrics import accuracy_score, f1_score, recall_score
import argparse
import json
import pandas as pd
import pdb
import numpy as np
from tqdm import tqdm

# file arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--predictions", required=True, help="file path to all the bootstrap_predictions of the test set") # this must be a list of predictions in the form of json
parser.add_argument("-t", "--truths", required=True, help="file path to the csv file with the true bootstrap_truths") # this must be in the csv file used for training in a column called BiasLabel
parser.add_argument("-o", "--output", required=False, help="path to folder to store bootstrapping results", default=".") # this is the directory where you store the final results
parser.add_argument("-n", "--number", required=False, help="number of bootstraps for estimation", type=int, default=1000) # this is the number of bootstraps to perform
parser.add_argument("-i", "--indices", required=False, help="file path to json with the indices to pull from", default=None) # this parameter is to select a sample of predictions that we use to score the models against the "challenging subset" (i.e. notes with stigmatizing terms but could be used in a stigmatizing or non-stigmatizing context)

if __name__ == "__main__":
    args = parser.parse_args()

    # loading the predictions json file
    predictions = json.load(open(args.predictions))
    # loading the true labels
    truths = list(pd.read_csv(args.truths)["BiasLabel"])
    # these are the unique identifiers for the data values (to group notes from the same encounter within one data split)
    note_ids = list(pd.read_csv(args.truths)["event_id"])
    if args.indices != None:
        # if requested, extract the unique identifiers that fall under the subset of data you want to evaluate
        selected_indices = json.load(open(args.indices))
        # load the predictions and the truth values for that subset
        truths = [label for i, label in zip(note_ids, truths) if i in selected_indices]
        predictions = [p for i, p in zip(note_ids, predictions) if i in selected_indices]
    # create an index filter list after selecting for indices if there were any; otherwise the indices will just be all datapoints
    indices = list(range(0, len(predictions)))

    bootstraps = []
    # iterate an n number of bootstraps
    for _ in tqdm(range(args.number)):
        # randomly select with replacement "new datasets" of the same size from the pool sampled to create a bootstrap 
        bootstrap_indices = np.random.choice(indices, len(indices), replace=True)
        # select the predictions and truths from the bootstrap generated
        bootstrap_predictions = np.array(predictions)[bootstrap_indices]
        bootstrap_truths = np.array(truths)[bootstrap_indices]
        
        # scores generated using sci-kit learn scoring functions
        acc = accuracy_score(bootstrap_truths, bootstrap_predictions)
        f1_pos = f1_score(bootstrap_truths, bootstrap_predictions, pos_label=1)
        f1_neg = f1_score(bootstrap_truths, bootstrap_predictions, pos_label=0)
        f1_micro = f1_score(bootstrap_truths, bootstrap_predictions, average="micro")
        f1_macro = f1_score(bootstrap_truths, bootstrap_predictions, average="macro")
        f1_weighted = f1_score(bootstrap_truths, bootstrap_predictions, average="weighted")
        sensitivity = recall_score(bootstrap_truths, bootstrap_predictions, pos_label=1)
        specificity = recall_score(bootstrap_truths, bootstrap_predictions, pos_label=0)
        bootstrap_temp = [acc, f1_pos, f1_neg, f1_micro, f1_macro, f1_weighted, sensitivity, specificity]
        
        # store the bootstraps
        bootstraps.append(bootstrap_temp)
    
    # calculate mean and std for the generation of confidence intervals for reporting later
    arr = np.array(bootstraps)
    means = list(np.mean(arr, axis=0))
    sds = list(np.std(arr, axis=0))
    json.dump({"means": means, "standard_deviations": sds}, open(args.output, "w"))