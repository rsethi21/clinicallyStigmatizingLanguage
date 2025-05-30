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


## Run Inference
- Zeroshot Approach
```
nohup python3 path/to/inference.py -p path/to/zeroshot.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- Retrieval Augmented Generation (RAG) Approach
```
nohup python3 path/to/inference.py -p path/to/rag.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- In-Context Approach
```
nohup python3 path/to/inference.py -p path/to/context.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- SFT Approach
```
nohup python3 path/to/inference.py -p path/to/sft.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- Baseline Keyword Approach
```
python3 path/to/manual.py -p path/to/manual.yml -d path/to/csv/file -o path/to/output/json
```

## Score
```
python3 bootstrapping.py -p path/to/output/json/from/model -t path/to/csv/file/inputed/to/model -o path/to/save/output/json -n number/of/bootstraps
```

## Data
- Store clinical notes to evaluate under the header "Assessment"
- Store labels for notes under the header "BiasLabel"

## Notes
- Scripts to multiprocess are provided but not necessary for reproducing results
- Scripts to process data (splitting, preventing data leakage) are provided but not necessary to reproduce results
