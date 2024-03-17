from langchain.chat_models import ChatGooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import gradio as gr
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import time
PATH = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/db_pdf/operating_systems_three_easy_pieces"
PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"
GOOGLE_API_KEY_PATH = os.path.join(PATH_DATA, "google_api.txt")
OPENAI_API_KEY_PATH = os.path.join(PATH_DATA, "openai_key.txt")
def load_api_keys():
    with open(GOOGLE_API_KEY_PATH, "r") as f:
        google_api_key = f.read().strip()
    with open(OPENAI_API_KEY_PATH, "r") as f:
        openai_api_key = f.read().strip()
    return google_api_key, openai_api_key
GOOGLE_API_KEY, OPENAI_API_KEY = load_api_keys()
llm = ChatGooglePalm(google_api_key=GOOGLE_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vs_doc = FAISS.load_local(PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vs_doc.as_retriever(search_type="mmr", search_kwargs=dict(k=3))

# Initialize memories
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="chat_history")
memory2 = ConversationBufferMemory(memory_key="chat_history")
def response(message, history, additional_input_info):
    list_skip = ["summary", "summarize", "Summary", "Summarize"]
    
    # Choose memory based on query content
    tmp_memory = memory2 if any(skip in message for skip in list_skip) else memory
    
    prompt_template = """You are a nice chatbot having a conversation with a human.
    Previous conversation:
    {chat_history}
    New human question: {question}. Please answer this question, answer as simple as possible. Answer precisely.
    Response:"""
    
    conversation = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template, question=message),
        verbose=True,
        memory=tmp_memory
    )

    try:
        output = conversation.invoke(message)
        response_text = output['text']
        memory2.save_context({"input": message}, {"output": response_text})
    except Exception as e:
        response_text = "Okay, got your idea. Go ahead."
        print(e)

    return response_text


# Gradio interface setup
gr.ChatInterface(
    fn=response,
    textbox=gr.Textbox(placeholder="Talk to me..", container=False, scale=7),
    chatbot=gr.Chatbot(height=1000),
    title="What chatbot do you want?",
    description="Ask and I shall respond.",
    theme="Monochrome", #"soft"
    examples=[["It's hot today :( "], ["Lunch menu suggestions, choose between noodles or rice"], ["Tell me about A15 bionic chipset."]],
    retry_btn="Retry ↩",
    undo_btn="Delete Last Chat ❌",
    clear_btn="Clear All Chats 💫",
).launch()