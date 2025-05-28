import pandas as pd
import re

if __name__ == "__main__":
    path = "inputs/raw_data/all.csv"
    df = pd.read_csv(path)
    for i, data in df.iterrows():
        subject_id = str(df.loc[i, "enc_id"])[0:7]
        search = "[A-Za-z ]+:"
        entry = open("inputs/raw_data/mimiciii_hermes/"+data.FilePath).read()
        headers = re.findall(search, entry)
        values = re.split(search, entry)
        print(headers)
        exit()