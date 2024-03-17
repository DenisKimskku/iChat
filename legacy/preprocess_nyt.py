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
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAI
import warnings
import time
import random
from tqdm import tqdm
start_time = time.time()
warnings.simplefilter(action='ignore')

PATH = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/db_faiss_nyt"
PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"
#GOOGLE_API_KEY is on PATH_DATA+"google_api.txt". Read from there
with open(PATH_DATA+"google_api.txt", "r") as f:
    GOOGLE_API_KEY = f.read()
GOOGLE_API_KEY = GOOGLE_API_KEY.strip()
with open(PATH_DATA + "openai_key.txt", "r") as f:
    OPENAI_API_KEY = f.read()
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
# Set the random seed for reproducibility
torch.manual_seed(42)
import os
#'''



#'''
from langchain_community.vectorstores import FAISS
if not os.path.exists(PATH + "/index.faiss"):
    #define loader using langchain
    loader = TextLoader(PATH_DATA + "nyt_10.txt")
    #load the document
    document = loader.load()
    #print(document)
    #print number of documents
    print(f"Number of documents: {len(document)}")
    #random shuffle the document, and get the first 20% of the document
    #random.shuffle(document)
    #document = document[:int(len(document)*0.2)]

    #split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    text_split = splitter.split_documents(document)
    #print(text_split)
    print(f"Number of chunks: {len(text_split)}")
    
    ch_time = time.time()
    #vs_doc = Chroma.from_documents(documents = text_split, embedding = embeddings, persist_directory = PATH)
    #vs_doc = FAISS.from_documents(documents = text_split, embedding = embeddings)
    #'''
    vs_doc = None
    
    with tqdm(total=len(text_split), desc="Ingesting documents") as pbar:
        for i, doc in enumerate(text_split):
            if i == 0:
                vs_doc = FAISS.from_documents(documents = [doc], embedding = embeddings)
            else:
                try:
                    vs_doc_ingest = FAISS.from_documents(documents = [doc], embedding = embeddings)
                except Exception as e:
                    print(e)
                    time.sleep(0.06)
                    continue
                vs_doc.merge_from(vs_doc_ingest)
                #print(f"Number of documents: {vs_doc}")
            pbar.update(1)
    #'''
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
vs_doc = FAISS.load_local(PATH, embeddings, allow_dangerous_deserialization = True)
retriever = vs_doc.as_retriever(search_type="mmr", search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="chat_history")
memory2 = ConversationBufferMemory(memory_key="chat_history")#, return_docs = False)
epoch = 5
#load the vectorstore
for cnt in range(1, epoch+1):
    template2 = """You are a nice chatbot having a conversation with a human.
    Previous conversation:
    {chat_history}
    New human question: {question} . Please answer this question, answer as simple as possible. Answer precisely.
    Response:"""
    prompt2 = PromptTemplate.from_template(template2)
    print(f"Epoch {cnt}")
    query = input("Enter your query: ")
    query = query.strip()
    #does query contains "summary  " or "summarize", need to use memory2 instead of memory
    tmp_memory = memory
    #llm = ChatGooglePalm(google_api_key=GOOGLE_API_KEY)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-instruct")
    tim1 = time.time()
    print(f"Time taken: {time.time()-tim1:.2f}s")
    list_skip = ["summary", "summarize", "Summary", "Summarize"]
    if any(skip in query for skip in list_skip):
        # Define the chain
        print("Using memory2")
        conversation = LLMChain(
            llm=llm,
            prompt=prompt2,
            verbose=True,
            memory=memory2
        )
    else:
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
        print(e)
        output = "Okay, got your idea. Go ahead."
    #print(output)
    memory2.save_context({"input": query}, {"output": output['text']})
    print(output['text'])
    