import json
import pandas as pd
from tqdm import tqdm
import pdb
from transformers import AutoTokenizer

if __name__ == "__main__":
    data = json.load(open("outputs/output.json"))
    tokenizer = AutoTokenizer.from_pretrained("/home1/rsethi1/stigmatizing_lang_rsh/outputs/models/sft", local_files_only=True)

    ori_sentence = data[0]["examples"][0]
    ex_sentence = data[0]["examples"][2]
    tokens = tokenizer.encode(ex_sentence, add_special_tokens=False)
    try:
        tokens = tokens["input_ids"]
    except:
        pass
    reformatted_examples = [ori_sentence]
    split_by = int(0.05*len(tokens))
    split_by = 5

    for i in range(0, len(tokens), 1):
        new_input = tokenizer.decode(tokens[0:i])
        reformatted_examples.append(new_input)
    
    json.dump([{"examples": reformatted_examples, "original": ori_sentence}], open("outputs/iterative_train.json", "w"))