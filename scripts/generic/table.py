import os
import pandas as pd
import json

main_folder = "/home1/rsethi1/stigmatizing_lang_rsh/outputs/test_results"
folders = ["sft", "full", "rag", "zeroshot_chunking", "manual"]
labels = ["accuracy", "f1_pos", "f1_neg", "f1_micro", "f1_macro", "f1_weighted", "sensitivity", "specificity"]
dictionaries_means_all = []
dictionaries_sds_all = []
dictionaries_means_terms = []
dictionaries_sds_terms = []

for f in folders:
	folder = os.path.join(main_folder, f)
	file = os.path.join(folder, "bootstrapping_all.json")
	d = json.load(open(file, "r"))
	temp = {l: v for l, v in zip(labels, d["means"])}
	means = {l: v for l, v in zip(labels, d["means"])}
	temp = {}
	temp["experiment"] = f
	temp["terms"] = False
	for k, v, in means.items():
		temp[k] = v
	dictionaries_means_all.append(temp)
	means = {l: v for l, v in zip(labels, d["standard_deviations"])}
	temp = {}
	temp["experiment"] = f
	temp["terms"] = False
	for k, v, in means.items():
		temp[k] = v
	dictionaries_sds_all.append(temp)

for f in folders:
	folder = os.path.join(main_folder, f)
	file = os.path.join(folder, "bootstrapping_terms.json")
	d = json.load(open(file, "r"))
	temp = {l: v for l, v in zip(labels, d["means"])}
	means = {l: v for l, v in zip(labels, d["means"])}
	temp = {}
	temp["experiment"] = f
	temp["terms"] = True
	for k, v, in means.items():
		temp[k] = v
	dictionaries_means_terms.append(temp)
	means = {l: v for l, v in zip(labels, d["standard_deviations"])}
	temp = {}
	temp["experiment"] = f
	temp["terms"] = True
	for k, v, in means.items():
		temp[k] = v
	dictionaries_sds_terms.append(temp)

print()
dictionaries_means_all.extend(dictionaries_means_terms)
dictionaries_sds_all.extend(dictionaries_sds_terms)
final_means = pd.DataFrame(dictionaries_means_all)
final_sds = pd.DataFrame(dictionaries_sds_all)


print(dictionaries_sds_all)
# print(dictionaries_sds_all)