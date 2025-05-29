import pandas as pd
import numpy as np
import re
from collections import Counter
import os
import random
import argparse
from tqdm import tqdm
import pdb


parser = argparse.ArgumentParser()
# arguments to help split the data into train, test
parser.add_argument("-i", "--input", help="input to csv file with data", required=True) # path to all data values
parser.add_argument("-s", "--split", help="proportion for smaller set", required=False, type=float, default=0.3) # proportion of data for smaller set
parser.add_argument("-o", "--output", help="output path", required=False, default=".") # folder to store output files to
parser.add_argument("-b", "--balance", help="boolean to create balanced set of 2 label dataset", required=False, type=bool, default=False) # boolean flag to split data while balancing the proportion of the labels
parser.add_argument("-r", "--ratio_stratify", help="boolean to stratify for 2 label dataset", required=False, type=bool, default=True) # boolean flag to split data while preserving the existing proportion of labels
parser.add_argument("-f", "--folder", help="path to all files", required=True) # path to folder with all the assessment text files

if __name__ == "__main__":
    args = parser.parse_args()
    # open csv file
    df = pd.read_csv(args.input)
    
    # store assessments from the text files into this new csv
    df["Assessment"] = ["".join(list(open(os.path.join(args.folder, f), "r").readlines())) for f in list(df.FilePath)]

    # identify unique encounter ids created by the combine admissions script; basically identifying the unique encounters for later in the script
    counts = dict(Counter(df.enc_id))
    keys = list(counts.keys())
    random.shuffle(keys) # shuffle to reduce sampling bias

    # in either split strategy, storing labels for the unique encounter keys
    if args.balance or args.ratio_stratify:
        labels = []
        for s in keys:
            labels.append(df.loc[np.where(np.array(df.enc_id) == s)[0][0], "BiasLabel"])

    # determining minimum size for test set using proportion
    target = int(args.split*len(df.index))

    # check to see if split size to balance per label is too big for the actual available entries; if so set as minimum
    if args.balance:
        split_size = target/2
        if split_size > len(np.where(df.BiasLabel == 0)[0]) or split_size > len(np.where(df.BiasLabel == 1)[0]):
            split_size = min(len(np.where(df.BiasLabel == 0)[0]), len(np.where(df.BiasLabel == 1)[0]))

    # add values to smaller set
    val_set_sentences = []
    if args.balance:
        # keep track of how many entries from each label are found
        current_neg = 0
        current_pos = 0
        # iterate the unique encounter ids generated for positive and negative labels seaprately
        # we add the entries that have the same encoutner id and keep track of how many added until split size requirement is met
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
    # if stratified ratio, the logic is still relevant from the top just different split targets based on natural prevelance of each label
    elif args.ratio_stratify:
        current_neg = 0
        current_pos = 0
        negative_split = int(args.split*len(df[df.BiasLabel == 0]))
        positive_split = int(args.split*len(df[df.BiasLabel == 1]))
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
    # if no stratification or balancing mentioned just add randomly
    else:
        current = 0
        for sentence in keys:
            val_set_sentences.append(sentence)
            current += counts[sentence]
            if current >= target:
                break
    # those added to the set above are then removed from the other split
    indices = df.index[df.enc_id.isin(val_set_sentences)]
    val_df = df.iloc[indices,:]
    dev_df = df.iloc[~df.index.isin(indices),:]
    val_df.reset_index(inplace=True)
    dev_df.reset_index(inplace=True)
    val_df.drop(columns=["index"], inplace=True)
    dev_df.drop(columns=["index"], inplace=True)
    # save (currently manually have to change the titles of the sets but not the folders)
    val_df.to_csv(os.path.join(args.output, "test.csv"), index=False)
    dev_df.to_csv(os.path.join(args.output, "train.csv"), index=False)