import pandas as pd
import numpy as np
import json

df_path = "/home1/rsethi1/stigmatizing_lang_rsh/inputs/raw_data/test.csv"
predictions_path = "/home1/rsethi1/stigmatizing_lang_rsh/outputs/test_results/manual/generated_outputs.json"
n = 25
out_path = "manual_error_analysis.csv"
random_state = 0

df = pd.read_csv(df_path)
predictions = json.load(open(predictions_path))
labels = df.BiasLabel

indices = np.where(~(predictions == labels))[0]
np.random.seed(random_state)
np.random.shuffle(indices)
selected_is = indices[0:n]

assessments = df.Assessment[selected_is]
selected_ps = np.array(predictions)[selected_is]
selected_ls = labels[selected_is]

out = pd.DataFrame.from_dict({"prediction": selected_ps, "label": selected_ls, "assm": assessments})
out.to_csv(out_path, index=False)

print(f"FP: {len(np.where(~(predictions == labels) & (np.array(predictions) == 1))[0])}")
print(f"FN: {len(np.where(~(predictions == labels) & (np.array(predictions) == 0))[0])}")