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

def login(email, password):
    if email == "admin" and password == "admin":
        return "Login successful"
    else:
        return "Login failed"

def register(email, password):
    if email == "admin1" and password == "admin1":
        return "Register successful"
    else:
        return "Register failed"
    

#Goal is: when register or login is successful, then chatbot interface is launched.
'''
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
'''
#Goal is: when register or login is successful, then chatbot interface is launched.

io1 = gr.Interface(fn=login, inputs=["text", "text"], outputs="text", title="Login", theme="Monochrome")
io2 = gr.ChatInterface(
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
)
'''
io1 = gr.Interface(lambda x:x, "textbox", "textbox")
io2 = gr.Interface(lambda x:x, "image", "image")

def show_row(value):
    if value=="Interface 1":
        return (gr.update(visible=True), gr.update(visible=False))  
    if value=="Interface 2":
        return (gr.update(visible=False), gr.update(visible=True))
    return (gr.update(visible=False), gr.update(visible=False))

with gr.Blocks() as demo:
    d = gr.Dropdown(["Interface 1", "Interface 2"])
    with gr.Row(visible=False) as r1:
        io1.render()
    with gr.Row(visible=False) as r2:
        io2.render()
    d.change(show_row, d, [r1, r2])
    
demo.launch()
'''


#use above template code to make a login interface, if login is successful, land to chatbot interface.
output = None 
def show_row(value):
    if value=="Login":
        output = io1.render()
    if value=="Register":
        output = io2.render()
    return output
with gr.Blocks() as demo:
    d = gr.Dropdown(["Login", "Register"])
    with gr.Row(visible=False) as r1:
        io1.render()
    with gr.Row(visible=False) as r2:
        io2.render()
    d.change(show_row, d, [r1, r2])
demo.launch()
