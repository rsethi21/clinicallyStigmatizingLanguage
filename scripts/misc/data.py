#!/usr/bin/env python3

import os, sys, pandas, datasets, json, re
from rouge_score import rouge_scorer
import yaml

def readYaml(fp):
  with open(fp, "r") as file:
      dictionary = yaml.safe_load(file)
  return dictionary

def calc_rougel(generated_text, reference_text):
  """Compute Rouge-L score"""

  # {'rougeL': Score(precision=0.5, recall=0.6, fmeasure=0.5)}
  scorer = rouge_scorer.RougeScorer(['rougeL'])
  scores = scorer.score(reference_text, generated_text)
  f1 = scores['rougeL'].fmeasure

  return f1

def csv_to_json(input_csv_path, output_json_path, system_prompt):
  """Convert to json to use for SFT"""

  df = pandas.read_csv(input_csv_path, dtype='str')

  # list of dictionaries to save as json
  samples = []

  for assm, summ, in zip(df['Assessment'], df['Summary']):
    if type(assm) == str and type(summ) == str:
      summ = summ.replace('#', '') # cleanup
      summ = summ.replace(':', '') # cleanup

      sample = {'instruction': system_prompt,
                'input': assm,
                'output': summ}
      samples.append(sample)

  json_file = open(output_json_path, 'w')
  json.dump(samples, json_file, indent=2)

def probing_inputs(input_to_yml):
  inputs = readYaml(input_to_yml)
  processed_inputs = []
  for question in inputs["sample_prompts"]:
    processed_inputs.append(f"[INST]\n<<SYS>>\n{inputs['system_prompt']}\n<</SYS>>\n{question}\n[/INST]\n\n")
  return processed_inputs

def pretraining_format(input_to_csv, output_path):
  entries = pandas.read_csv(input_to_csv)
  finalized_list = []
  with open(output_path, "w") as out_file:
    for entry in entries.Assessment:
      try:
        e = entry.replace('\n',' ')
        e = re.sub('\s+', ' ', entry)
        out_file.write(e[:len(e)-1] + "\n")
      except:
        pass