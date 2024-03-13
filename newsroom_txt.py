PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/" + "release/train.jsonl"
#this is making the newsroom.txt file. Extract the "text" field in jsonl file, and save it to newsroom.txt. Progress bar is also shown. Only 512 token is saved.
import json
import os
from tqdm import tqdm

with open(PATH_DATA, "r") as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
len_data = len(data)
len_data=  len_data * 0.1
data = data[:int(len_data)]
with open(PATH_DATA.replace("train.jsonl", "newsroom_10.txt"), "w") as f:
    for d in tqdm(data):
        f.write("%s\n" % d["text"][:512])
        
