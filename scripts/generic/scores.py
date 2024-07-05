from sklearn.metrics import accuracy_score, f1_score, recall_score
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label", help="csv with label column", required=True)
parser.add_argument("-p", "--predictions", help="json file with predictions", required=True)
parser.add_argument("-o", "--output", help="path to save json file of scores", required=False, default=".")

if __name__ == "__main__":
    args = parser.parse_args()
    labels = list(pd.read_csv(args.label)["Label"])
    predictions = json.load(open(args.predictions))
    acc = accuracy_score(labels, predictions)
    f1_pos = f1_score(labels, predictions, pos_label=1)
    f1_neg = f1_score(labels, predictions, pos_label=0)
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    sensitivity = recall_score(labels, predictions, pos_label=1)
    specificity = recall_score(labels, predictions, pos_label=0)
    dictionary = {"accuracy": acc, "f1_pos": f1_pos, "f1_neg": f1_neg, "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted, "sensitivity": sensitivity, "specificity": specificity}
    json.dump(dictionary, open(args.output, "w"))