from tqdm import tqdm
import argparse
import pandas as pd
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor as ppe

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--admissions", help="path with admissions information", required=True)
parser.add_argument("-d", "--data", help="path with data", required=True)
parser.add_argument("-n", "--num", help="number of processors to use", required=False, type=int, default=1)

def encounter(admissions, note_id):
    admission_id = admissions[admissions.ROW_ID == note_id]["HADM_ID"].values[0]
    cgid = admissions[admissions.ROW_ID == note_id]["CGID"].values[0]
    subject_id = admissions[admissions.ROW_ID == note_id]["SUBJECT_ID"].values[0]
    return f"{subject_id}-{admission_id}-{cgid}"

if __name__ == "__main__":
    args = parser.parse_args()
    admissions = pd.read_csv(args.admissions)
    data = pd.read_csv(args.data)

    note_ids = [note.split("/")[-1].split("_")[-1] for note in list(data.FilePath)]
    note_ids = [int(note[:note.index(".")]) for note in note_ids]
    data["event_id"] = note_ids

    with ppe(max_workers=args.num) as executor:
        enc_ids = list(tqdm(executor.map(encounter, repeat(admissions), note_ids), total=len(note_ids)))
    
    data["enc_id"] = enc_ids
    data.to_csv(args.data, index=False)