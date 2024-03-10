from langchain.chat_models import ChatGooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import gradio as gr
import os
import time
PATH = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/db_faiss"
PATH_DATA = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"
with open(os.path.join(PATH_DATA, "google_api.txt"), "r") as f:
    GOOGLE_API_KEY = f.read().strip()
llm = ChatGooglePalm(google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history")
def response(message, history, additional_input_info):
    prompt_template = """You are a nice chatbot having a conversation with a human.
    Previous conversation:
    {chat_history}
    New human question: {message} . Please answer this question, answer as simple as possible. Answer precisely.
    Response:"""
    
    # Handling the conversation
    conversation = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template),
        verbose=True,
        memory=memory
    )
    try:
        output = conversation.invoke(message)
        response_text = output['text']
    except Exception as e:
        response_text = "Okay, got your idea. Go ahead."

    return response_text

# Gradio interface setup
gr.ChatInterface(
    fn=response,
    textbox=gr.Textbox(placeholder="Talk to me..", container=False, scale=7),
    chatbot=gr.Chatbot(height=1000),
    title="What chatbot do you want?",
    description="Ask and I shall respond.",
    theme="soft",
    examples=[["It's hot today :( "], ["Lunch menu suggestions, choose between noodles or rice"], ["Tell me about A15 bionic chipset."]],
    retry_btn="Retry ↩",
    undo_btn="Delete Last Chat ❌",
    clear_btn="Clear All Chats 💫",
).launch()