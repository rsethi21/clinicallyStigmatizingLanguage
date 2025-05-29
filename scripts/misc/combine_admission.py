from tqdm import tqdm
import argparse
import pandas as pd
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor as ppe
import pdb

# preventing data leakage between splits
# need note events table and extracted assessments from our labelling process to map the correct ids
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="path with data", required=True) # data from labelling process
parser.add_argument("-e", "--events", help="path with note events", required=True) # data from note-events mimic iii

# function to create hybrid unique id to group notes
# take in note events and id for which we are matching the other ids to
def encounter(notes, note_id):
    admission_id = notes[notes.ROW_ID == note_id]["HADM_ID"].values[0]
    cgid = notes[notes.ROW_ID == note_id]["CGID"].values[0]
    subject_id = notes[notes.ROW_ID == note_id]["SUBJECT_ID"].values[0]
    return f"{subject_id}-{admission_id}-{cgid}"

if __name__ == "__main__":
    args = parser.parse_args()
    # load data requested
    data = pd.read_csv(args.data)
    events = pd.read_csv(args.events)

    # extract row ids from the file paths (we have file paths in the labeled dataset that contain the row id for the note events table)
    note_ids = [note.split("/")[-1].split("_")[-1] for note in list(data.FilePath)]
    note_ids = [int(note[:note.index(".")]) for note in note_ids]

    # store in dataset
    data["event_id"] = note_ids

    # encounter ids using the encounter function above
    enc_ids = []
    for note_id in tqdm(note_ids):
        enc_ids.append(encounter(events, note_id))
    
    # save file
    data["enc_id"] = enc_ids
    data.to_csv(args.data, index=False)