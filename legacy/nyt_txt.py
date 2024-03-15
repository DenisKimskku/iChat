PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/" + "nyt-metadata.csv"
#This is making nyt.txt file. Extract the "abstract" field in csv file, and save it to nyt.txt. Progress bar is also shown. Only 512 token is saved. If abstract field is null or 'To the Editor:.', it is not saved.
import pandas as pd
import os
from tqdm import tqdm
df = pd.read_csv(PATH_DATA)
df = df.dropna(subset=['abstract'])
df = df[df['abstract'] != 'To the Editor:']
#skip null
print(f"Number of documents: {len(df)}")
len_data = len(df)
#only last 10% of the data
len_data = len_data * 0.1
df = df[-int(len_data):]
with open(PATH_DATA.replace("nyt-metadata.csv", "nyt_10.txt"), "w") as f:
    for d in tqdm(df['abstract']):
        f.write("%s\n" % d[:512])