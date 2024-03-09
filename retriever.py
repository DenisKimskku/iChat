import warnings
warnings.filterwarnings("ignore")
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma


mps_device = torch.device('mps')
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map = "auto")
#model.to(mps_device)
time1 = time()
#input_text = "What do u think about the new iPhone?"
input_text = "Hi, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").to(mps_device)
outputs = model.generate(**input_ids,max_new_tokens=40)
time2 = time()
print(time2-time1)
#list of unusable tokens
list_tokens = ['<pad>','<unk>','<s>','</s>']
outputs = tokenizer.decode(outputs[0])
#remove the unusable tokens
for token in list_tokens:
    outputs = outputs.replace(token,'')
print(outputs)