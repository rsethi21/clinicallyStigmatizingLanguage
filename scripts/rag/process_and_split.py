import pandas as pd
import numpy as np
import re
from collections import Counter
import os
import random
import argparse
from tqdm import tqdm
import pdb

# add label balancing

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input to csv file with data", required=True)
parser.add_argument("-s", "--split", help="proportion for validation set", required=False, type=float, default=0.2)
parser.add_argument("-o", "--output", help="output path", required=False, default=".")
parser.add_argument("-b", "--balance", help="boolean to create balanced set of 2 label dataset", required=False, type=bool, default=False)
parser.add_argument("-r", "--ratio_stratify", help="boolean to stratify for 2 label dataset", required=False, type=bool, default=True)
parser.add_argument("-f", "--folder", help="path to all files", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    identified_sentences = list(df.IdentifiedSentence)
    for i, sentence in enumerate(identified_sentences):
        identified_sentences[i] = re.sub(r"\s+", " ", sentence)
    df.IdentifiedSentence = identified_sentences
    counts = dict(Counter(identified_sentences))
    keys = list(counts.keys())
    random.shuffle(keys)
    if args.balance or args.ratio_stratify:
        labels = [df.loc[np.where(np.array(df.IdentifiedSentence) == s)[0][0], "Label"] for s in keys]
    target = int(args.split*len(identified_sentences))
    if args.balance:
        split_size = target/2
        if split_size > len(np.where(df.Label == 0)[0]) or split_size > len(np.where(df.Label == 1)[0]):
            split_size = min(len(np.where(df.Label == 0)[0]), len(np.where(df.Label == 1)[0]))
    val_set_sentences = []
    if args.balance:
        current_neg = 0
        current_pos = 0
        for sentence,label in zip(keys,labels):
            if label == 1:
                val_set_sentences.append(sentence)
                current_pos += counts[sentence]
            if current_pos >= split_size:
                break
        for sentence,label in zip(keys,labels):
            if label == 0:
                val_set_sentences.append(sentence)
                current_neg += counts[sentence]
            if current_neg >= split_size:
                break
    elif args.ratio_stratify:
        current_neg = 0
        current_pos = 0
        negative_split = int(args.split*len(df[df.Label == 0]))
        positive_split = int(args.split*len(df[df.Label == 1]))
        for sentence,label in zip(keys,labels):
            if label == 1:
                val_set_sentences.append(sentence)
                current_pos += counts[sentence]
            if current_pos >= positive_split:
                break
        for sentence,label in zip(keys,labels):
            if label == 0:
                val_set_sentences.append(sentence)
                current_neg += counts[sentence]
            if current_neg >= negative_split:
                break
    else:
        current = 0
        for sentence in keys:
            val_set_sentences.append(sentence)
            current += counts[sentence]
            if current >= target:
                break
    indices = df.index[df.IdentifiedSentence.isin(val_set_sentences)]
    val_df = df.iloc[indices,:]
    dev_df = df.iloc[~df.index.isin(indices),:]
    val_df["Assessment"] = ["".join(list(open(os.path.join(args.folder, f), "r").readlines())) for f in list(val_df.NoteID)]
    dev_df["Assessment"] = ["".join(list(open(os.path.join(args.folder, f), "r").readlines())) for f in list(dev_df.NoteID)]
    val_df.reset_index(inplace=True)
    dev_df.reset_index(inplace=True)
    val_df.drop(columns=["index"], inplace=True)
    dev_df.drop(columns=["index"], inplace=True)
    val_df.to_csv(os.path.join(args.output, "rag_val_3.csv"), index=True)
    dev_df.to_csv(os.path.join(args.output, "rag_test_3.csv"), index=True)