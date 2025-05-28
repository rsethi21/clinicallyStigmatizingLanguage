from sklearn.metrics import accuracy_score, f1_score, recall_score
import argparse
import json
import pandas as pd
import pdb
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--predictions", required=True, help="file path to all the bootstrap_predictions of the test set")
parser.add_argument("-t", "--truths", required=True, help="file path to the csv file with the true bootstrap_truths")
parser.add_argument("-o", "--output", required=False, help="path to folder to store bootstrapping results", default=".")
parser.add_argument("-n", "--number", required=False, help="number of bootstraps for estimation", type=int, default=1000)
parser.add_argument("-i", "--indices", required=False, help="file path to json with the indices to pull from", default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    predictions = json.load(open(args.predictions))
    truths = list(pd.read_csv(args.truths)["BiasLabel"])
    note_ids = list(pd.read_csv(args.truths)["event_id"])
    if args.indices != None:
        selected_indices = json.load(open(args.indices))
        truths = [label for i, label in zip(note_ids, truths) if i in selected_indices]
        predictions = [p for i, p in zip(note_ids, predictions) if i in selected_indices]
    indices = list(range(0, len(predictions)))

    bootstraps = []
    for _ in tqdm(range(args.number)):
        bootstrap_indices = np.random.choice(indices, len(indices), replace=True)
        bootstrap_predictions = np.array(predictions)[bootstrap_indices]
        bootstrap_truths = np.array(truths)[bootstrap_indices]
        
        acc = accuracy_score(bootstrap_truths, bootstrap_predictions)
        f1_pos = f1_score(bootstrap_truths, bootstrap_predictions, pos_label=1)
        f1_neg = f1_score(bootstrap_truths, bootstrap_predictions, pos_label=0)
        f1_micro = f1_score(bootstrap_truths, bootstrap_predictions, average="micro")
        f1_macro = f1_score(bootstrap_truths, bootstrap_predictions, average="macro")
        f1_weighted = f1_score(bootstrap_truths, bootstrap_predictions, average="weighted")
        sensitivity = recall_score(bootstrap_truths, bootstrap_predictions, pos_label=1)
        specificity = recall_score(bootstrap_truths, bootstrap_predictions, pos_label=0)
        # dictionary = {"accuracy": acc, "f1_pos": f1_pos, "f1_neg": f1_neg, "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted, "sensitivity": sensitivity, "specificity": specificity}
        bootstrap_temp = [acc, f1_pos, f1_neg, f1_micro, f1_macro, f1_weighted, sensitivity, specificity]
        
        bootstraps.append(bootstrap_temp)
    
    arr = np.array(bootstraps)
    means = list(np.mean(arr, axis=0))
    sds = list(np.std(arr, axis=0))
    json.dump({"means": means, "standard_deviations": sds}, open(args.output, "w"))