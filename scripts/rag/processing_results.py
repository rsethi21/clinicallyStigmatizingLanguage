import argparse
import os
import json
import yaml
import pandas as pd

folder = "/home1/rsethi1/stigmatizing_lang_rsh/outputs/sft"
results = {'chunking': [], 'num_context': [], 'scoring': [], 'context_path': [], 'accuracy': [], 'f1_pos': [], 'f1_neg': [], 'f1_micro': [], 'f1_macro': [], 'f1_weighted': [], 'sensitivity': [], 'specificity': []}

scores = json.load(open(os.path.join(folder, "scores_initial.json")))
for key, value in scores.items():
    results[key].append(value)
results["num_context"].append(None)
results["context_path"].append(None)
results["scoring"].append(None)
results["chunking"].append(None)
results["experiment"] = ["sft"]

df = pd.DataFrame.from_dict(results)
cols = list(df.columns)
reversed(cols)
df = df.loc[:,cols]
df.to_csv("./sft.csv")