# clinicallyStigmatizingLanguage
## Set-up (on Linux below, please adjust for other OS)
- create virtual environment or other package management
```
python3 -m venv venv
```
- activate virtual environment or other package management
```
source venv/bin/activate
```
- install requirements using pip
```
pip3 install -r path/to/requirements.txt
```

## Configs
- All approaches below that use LLMs have an LLM configs portion in their yaml files
- Additional parameters specific to the approach is appeneded below
```
llm:
  do_sample: True
  top_k: 10
  num_return_sequences: 1
  temperature: 0.001
  max_length: 6144
  max_new_tokens: 6144
  length_penalty: 1.0
  repetition_penalty: 1.0
  num_beams: 1
```
- Zeroshot Approach Yaml
```
method:
  formatting: llama3 # formatting instructions (currently only take llama3 for parsing of output)
  model_path: path/to/local/model/instance
  context_path: path/to/context # if desired to give the model information about stigmatizing language
  # system prompt
  sp: You are a physician revising your clinical notes. You must determine if your note has any clinically stigmatizing languange. Please answer simply yes or no and no other text.
  num_context: null # not needed for zeroshot
  scoring: null # not needed for zeroshot
  chunking: # chunking if desired to process notes as pieces; can set null
    chunk_size: 1000
    overlap_size: 100
```
- Retrieval Augmented Generation (RAG) Approach Yaml
```
method:
  chunking: # chunking if desired to process notes as pieces; can set null
    chunk_size: 1000
    overlap_size: 100
  context_path: path/to/context # requried for RAG (csv file with documents/text related to the task)
  formatting: llama3 # formatting instructions
  model_path: path/to/local/model/instance
  num_context: 5 # number of context entries (top n context entries will be selected based on scoring metric)
  scoring: dot # scoring metric
  # system prompt
  sp: You are a physician revising your clinical notes. You must determine if your note has any clinically stigmatizing language in terms of substance use. Please answer simply yes or no and no other text.
  transform: false # sequential prompting
  threshold: null # score threshold
```
- In-Context Approach Yaml
```
method:
  chunking: # chunking if desired to process notes as pieces; can set null
    chunk_size: 1000
    overlap_size: 100
  context_path: path/to/context # requried for In-Context (csv file with documents/text related to the task)
  formatting: llama3 # formatting instructions
  model_path: path/to/local/model/instance
  num_context: null # this should be null here to instruct the script to use all context entries
  scoring: null # this should be null here to instruct the script to use all context entries
  # system prompt
  sp: You are a physician revising your clinical notes. You must determine if your note has any clinically stigmatizing language in terms of substance use. Please answer simply yes or no and no other text.
  transform: false # sequential prompting
  threshold: null # score threshold
```
- SFT Approach Yaml
```
method:
  model_path: path/to/local/model/instance
  # system prompt
  sp: You are a physician revising your clinical notes. You must determine if your note has any clinically stigmatizing language in terms of substance use. Please answer simply yes or no and no other text.
```
- Baseline Keyword Approach Yaml
```
stigmatizing_terms:
  - <term1>
```

## Run Inference
- Zeroshot Approach (using context-based.py script)
```
nohup python3 path/to/context-based.py -p path/to/zeroshot.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- Retrieval Augmented Generation (RAG) Approach (using context-based.py script)
```
nohup python3 path/to/context-based.py -p path/to/rag.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- In-Context Approach (using context-based.py script)
```
nohup python3 path/to/context-based.py -p path/to/context.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- SFT Approach (using inference.py script)
```
nohup python3 path/to/inference.py -p path/to/sft.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- Baseline Keyword Approach (using keyword.py script)
```
python3 path/to/keyword.py -p path/to/keyword.yml -d path/to/csv/file -o path/to/output/json
```

## Score
```
python3 path/to/bootstrapping.py -p path/to/output/json/from/model -t path/to/csv/file/inputed/to/model -o path/to/save/output/json -n number/of/bootstraps
```

## Run Fine-Tuning
- need a ds_config file which is shared in the configs directory
- refer to llama factory documentation to execute training process (https://github.com/hiyouga/LLaMA-Factory)
```
./path/to/training/shell/script
```

## Data
- Store clinical notes to evaluate under the header "Assessment"
- Store labels for notes under the header "BiasLabel"

## Notes
- Scripts to multiprocess are provided but not necessary for reproducing results
- Scripts to process data (splitting, preventing data leakage) are provided but not necessary to reproduce results
