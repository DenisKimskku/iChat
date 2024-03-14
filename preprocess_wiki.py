import torch

#use wikitext-2 dataset from torchtext, which is a popular dataset for language modeling. Load dataset
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext import vocab
from langchain import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatGooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import warnings
import time
import random
start_time = time.time()
warnings.simplefilter(action='ignore')

PATH = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/db_faiss"
PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"
#GOOGLE_API_KEY is on PATH_DATA+"google_api.txt". Read from there
with open(PATH_DATA+"google_api.txt", "r") as f:
    GOOGLE_API_KEY = f.read()
GOOGLE_API_KEY = GOOGLE_API_KEY.strip()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
# Set the random seed for reproducibility
torch.manual_seed(42)
import os
# Load the wikitext-103 dataset
from datasets import load_dataset
#find corpus_3k.txt
if not os.path.exists(PATH_DATA+"corpus_300k.txt"):
    print("corpus_300k.txt not found")
    #dataset = load_dataset("wikitext", "wikitext-2-v1")
    dataset = load_dataset("wikipedia", "20220301.en")
    train_dataset = dataset["train"]['text']
    print(f"Number of documents: {len(train_dataset)}")
    #split the dataset to 100000, randomize
    from tqdm import tqdm
    #slice dataset as max tokens is 512
    for i in tqdm(range(len(train_dataset))):
        train_dataset[i] = train_dataset[i][:512]

    random.shuffle(train_dataset)
    #truncate the dataset to 100000
    num_doc = 3000
    train_dataset = train_dataset[:num_doc]
    #truncate the dataset to max 512 tokens
    train_dataset = [doc[:512] for doc in train_dataset]
    #remove '\n' from the dataset
    train_dataset = [doc.replace('\n', ' ') for doc in train_dataset]
    end_time = time.time()
    print(f"Preprocess - Time taken: {end_time-start_time:.2f}s")
    #test_dataset = dataset["test"]
    #val_dataset = dataset["validation"]
    # Print the first few examples
    #use bm25 to rank the documents, when given query
    query = "Tell me about a components of a computer"
    import time
    start = time.time()
    #dataset is stated above, find in the train_dataset
    #define bm25
    from rank_bm25 import BM25Okapi
    import numpy as np
    # Tokenize the documents
    tokenized_corpus = [train_dataset[i].split(" ") for i in range(len(train_dataset))]
    bm25 = BM25Okapi(tokenized_corpus)
    # Get the top 5 documents
    doc_scores = bm25.get_scores(query.split(" "))
    top_5 = np.argsort(doc_scores)[::-1][:5]
    end = time.time()
    print(f"Time taken: {end-start:.2f}s")
    
    for i, doc in enumerate(top_5):
        print(f"Rank {i+1}:", train_dataset[int(doc)])
        
    #save to text file
    with open(PATH_DATA+'corpus_3k.txt', 'w') as f:
        for doc in train_dataset:
            f.write("%s\n" % doc)
    quit()
#'''



#'''
from langchain_community.vectorstores import FAISS
if not os.path.exists(PATH + "/index.faiss"):
    #define loader using langchain
    loader = TextLoader(PATH_DATA+"corpus_300k.txt")
    #load the document
    document = loader.load()
    #use google palm embeddings

    #split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_split = splitter.split_documents(document)

    ch_time = time.time()
    #vs_doc = Chroma.from_documents(documents = text_split, embedding = embeddings, persist_directory = PATH)
    vs_doc = FAISS.from_documents(documents = text_split, embedding = embeddings)
    vs_doc.save_local(PATH)
    print(f"Time taken: {time.time()-ch_time:.2f}s")
    #vs_doc.persist()
    #print("Vectorstore is persisted")
    #print("Time taken: ", time.time()-ch_time)

    #quit()

#Intended to be testing#
import google.generativeai as palm
palm.configure(api_key=GOOGLE_API_KEY)

models = [model for model in palm.list_models()]

for model in models:
  print(model.name)


from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory

#memory = ConversationBufferMemory(memory_key="chat_history")#, return_docs = False)
#memory.chat_memory.add_user_message("Hello")
epoch = 5
#load the vectorstore
for cnt in range(1, epoch+1):
    vs_doc = FAISS.load_local(PATH, embeddings, allow_dangerous_deserialization = True)
    retriever = vs_doc.as_retriever(search_kwargs=dict(k=1))
    memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="chat_history")
    template2 = """You are a nice chatbot having a conversation with a human.
    Previous conversation:
    {chat_history}
    New human question: {question} . Please answer this question, answer as simple as possible. Answer precisely.
    Response:"""
    prompt2 = PromptTemplate.from_template(template2)
    print(f"Epoch {cnt}")
    query = input("Enter your query: ")
    query = query.strip()
    
    tim1 = time.time()
    print(f"Time taken: {time.time()-tim1:.2f}s")
    
    llm = ChatGooglePalm(google_api_key=GOOGLE_API_KEY)
    chat_history_tuples = []
    for message in memory:
        chat_history_tuples.append((message[0], message[1]))
    # Define the chain
    conversation = LLMChain(
        llm=llm,
        prompt=prompt2,
        verbose=True,
        memory=memory
    )
    try:
        output = conversation.invoke(query)
    except Exception as e:
        #print(e)
        output = "Okay, got your idea. Go ahead."
    print(output['text'])
    