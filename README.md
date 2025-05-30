# clinicallyStigmatizingLanguage
## Set-up
- Install required packages on linux/mac
```
python3 -m venv venv
```
```
source venv/bin/activate
```
```
pip3 install -r path/to/requirements.txt
```

## Run Inference
- SFT Model
```
nohup python3 path/to/inference.py -p path/to/sft.yml -d path/to/csv/file -o path/to/output/json > path/to/log/file 2>&1 &
```
- Baseline
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
