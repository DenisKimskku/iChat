import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
import time
import argparse  # For command-line options

PATH = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/db"
PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"
GOOGLE_API_KEY_PATH = os.path.join(PATH_DATA, "google_api.txt")
OPENAI_API_KEY_PATH = os.path.join(PATH_DATA, "openai_key.txt")

def load_api_keys():
    with open(GOOGLE_API_KEY_PATH, "r") as f:
        google_api_key = f.read().strip()
    with open(OPENAI_API_KEY_PATH, "r") as f:
        openai_api_key = f.read().strip()
    return google_api_key, openai_api_key

def create_embeddings(api_key, model="text-embedding-3-small", source='openai'):
    if source == 'google':
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    else:
        return OpenAIEmbeddings(model=model, openai_api_key=api_key)
    
def process_wikipedia_data():
    dataset = load_dataset("wikipedia", "20220301.en")
    train_dataset = dataset["train"]['text'][:300000] # Sample for simplicity
    train_dataset = [doc.replace('\n', ' ').strip()[:512] for doc in train_dataset]
    random.shuffle(train_dataset)
    with open(os.path.join(PATH_DATA, "corpus_300k.txt"), 'w') as f:
        for doc in train_dataset:
            f.write("%s\n" % doc)

def process_nyt_data():
    if not os.path.exists(os.path.join(PATH_DATA, "nyt-metadata.csv")):
        print("NYT metadata file not found.")
        #do this command: kaggle datasets download -d aryansingh0909/nyt-articles-21m-2000-present
        os.system("kaggle datasets download -d aryansingh0909/nyt-articles-21m-2000-present")
        os.system("unzip nyt-articles-21m-2000-present.zip")
        os.system("rm nyt-articles-21m-2000-present.zip")
        os.system("mv nyt-metadata.csv "+PATH_DATA)
    df = pd.read_csv(os.path.join(PATH_DATA, "nyt-metadata.csv"))
    df = df.dropna(subset=['abstract'])
    df = df[df['abstract'] != 'To the Editor:']
    df = df.tail(int(len(df) * 0.1))  # Last 10%
    with open(os.path.join(PATH_DATA, "nyt_10.txt"), 'w') as f:
        for abstract in df['abstract']:
            f.write("%s\n" % abstract[:512])


def process_pdf_data(pdf_path, embeddings, chunk_size, chunk_overlap, index_path):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    print(f"Number of documents: {len(document)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_split = splitter.split_documents(document)
    print(f"Number of chunks: {len(text_split)}")
    convert_to_faiss(text_split, embeddings, index_path)

def convert_to_faiss(text_split, embeddings, index_path):
    vs_doc = None
    for i, doc in enumerate(tqdm(text_split, desc="Processing documents")):
        if i == 0:
            vs_doc = FAISS.from_documents(documents=[doc], embedding=embeddings)
        else:
            try:
                vs_doc_ingest = FAISS.from_documents(documents=[doc], embedding=embeddings)
                vs_doc.merge_from(vs_doc_ingest)
            except Exception as e:
                print(e)
                time.sleep(0.06)
    vs_doc.save_local(index_path)

def main(dataset, pdf_path):
    print("Processing data...")
    print(f"Dataset: {dataset}")
    print(f"PDF path: {pdf_path}")
    random.seed(42)
    google_api_key, openai_api_key = load_api_keys()
    
    if dataset == 'wikipedia':
        if os.path.exists(os.path.join(PATH, "wikipedia/index.faiss")):
            print("Wikipedia data has already been processed.")
            return
        process_wikipedia_data()
        # Assuming Google embeddings are used for Wikipedia
        embeddings = create_embeddings(google_api_key, source='google')
        convert_to_faiss(os.path.join(PATH_DATA, "corpus_300k.txt"), embeddings, 1000, 0, os.path.join(PATH, "wikipedia"))
        
    elif dataset == 'nyt':
        if os.path.exists(os.path.join(PATH, "nyt/index.faiss")):
            print("NYT data has already been processed.")
            return
        process_nyt_data()
        # Assuming OpenAI embeddings are used for NYT
        embeddings = create_embeddings(openai_api_key)
        convert_to_faiss(os.path.join(PATH_DATA, "nyt_10.txt"), embeddings, 250, 0, os.path.join(PATH, "nyt"))
        
    elif dataset == 'pdf':
        name = pdf_path.split('/')[-1]
        if not os.path.exists(pdf_path):
            print("File does not exist.")
            return
        if os.path.exists(os.path.join(PATH, name.split('.')[0] + "/index.faiss")):
            print("PDF data has already been processed.")
            return
        embeddings = create_embeddings(openai_api_key)
        name = name.split('.')[0]
        process_pdf_data(pdf_path, embeddings, 1000, 0, os.path.join(PATH, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data to create FAISS embeddings.")
    parser.add_argument("--dataset", type=str, choices=['wikipedia', 'nyt', 'pdf'], help="The dataset to process.")
    parser.add_argument("--filepath", type=str, help="Path to the PDF file to process.", default="")
    args = parser.parse_args()

    main(args.dataset, args.filepath)  # Ensure your main function can handle this argument