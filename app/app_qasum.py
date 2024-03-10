import streamlit as st
import torch
import io
import base64
from datetime import datetime
import pypdf
import os
from llama_cpp import Llama
import pydantic
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from tempfile import NamedTemporaryFile
from langchain.vectorstores import Chroma, FAISS
import json
from langchain.chains.summarize import load_summarize_chain
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
import time
from langchain_community.document_loaders import WebBaseLoader

# summary_enable = True

#Title
st.set_page_config(page_title=" Jack of All Trades - 百事通")

from module import llm, pdf_processor

#load LLM
llm_chat_model = llm.load_chat_model()
summ_stuff_chain = load_summarize_chain(llm_chat_model, chain_type = "stuff", prompt = llm.sum_prompt_template)
    

# from tts_vits import tts
msg_template = [
      {
          "role": "system", 
           "content": "You are a teach assistant"
      },
      {
          "role": "user",
          "content": "",
      }
  ]

def index_builder(docs, is_vectordb = False):
    if is_vectordb:
        pass
    else:
        vdb_indexer = TFIDFRetriever.from_documents(docs, k = 4)
    return vdb_indexer

# Hugging Face Credentials
# Store store history
user_img_file = "./resource/cartoon.png"
sys_img_file = "./resource/logo.jpeg"

with st.sidebar:
    st.image(sys_img_file, caption = "Chat with Your Document")
    st.markdown('Answer what you know from provided document')

summary_enable = st.checkbox("Enable summary", value=False)
print(f"*** summary_enable = {summary_enable}")

doc_source = st.selectbox("Select document source", ["url", "pdf"])
#
if doc_source == "pdf":
    doc_file = st.file_uploader(
        label = "Chat with PDF Upload", 
        type = ["pdf"], 
        accept_multiple_files = False, 
        help = "upload pdf for summary & ask me anything in the document", 
        disabled = False
    )
    if doc_file:
        try:
            doc_content = pdf_processor.load_pdf(doc_file)
            max_page_limit = 2
            doc_content = doc_content[:max_page_limit]
        except:
            print(f"*** Failed loading {doc_file}")
            doc_content = None
elif doc_source == "url":
    doc_url = st.text_input("Document URL: ")
    try:
        doc_content = WebBaseLoader(doc_url).load()
    except:
        print(f"*** Failed loading {doc_url}")
        doc_content = None
else:
    doc_content = None

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Welcome to leaning course", 
            "dt": datetime.now().strftime("%H:%M:%S"),
            "avatar": sys_img_file,
            "feedback": "",
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    # print(message)
    with st.chat_message(message["role"], avatar = message["avatar"]):
        if message["role"] == "assistant":
            if message["feedback"] is not None:
                st.write(":orange[" + message["dt"] + "]\t" + message["feedback"])
            else:
                st.write(":orange[" + message["dt"] + "]\t")
        else:
            if message["content"] is not None:
                st.write(":orange[" + message["dt"] + "]\t" + message["content"])
            else:
                st.write(":orange[" + message["dt"] + "]\t")
        
#chat counter
if "wav_counter" not in  st.session_state:
    st.session_state["wav_counter"] = 0
else:
    st.session_state["wav_counter"] += 1

if doc_content:
    # doc_content = pdf_processor.load_pdf(doc_file)
    # print(f"***111** doc_content = {llm_chat_model} {doc_content}")
    if summary_enable:
        with st.container():
            with st.spinner("Working on generating summary ..."):        
                output_summ = summ_stuff_chain.run(doc_content)
                # st.write("SUMMARY: \n" + output_summ)
                st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]" + "\tSUMMARY: " + output_summ)
                message = {
                    "role": "assistant", 
                    "content": "Write summary about the document.", 
                    "dt": datetime.now().strftime("%H:%M:%S"),
                    "avatar": sys_img_file,
                    "feedback": output_summ,
                }
                st.session_state.messages.append(message)

    #Q1
    db_index = index_builder(doc_content, is_vectordb = False)
    qa_stuff_chain = load_qa_chain(llm_chat_model, chain_type="stuff", prompt = llm.qa_prompt_template)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append(
        {
            "role": "user", 
            "content": prompt, 
            "dt": datetime.now().strftime("%H:%M:%S"),
            "avatar": user_img_file,
            "feedback": "",
        }
    )
    with st.chat_message("user", avatar=user_img_file):
        st.write(":orange[" + datetime.now().strftime("%H:%M:%S") + "]\t" + prompt)

# TTS response
prompt_feedback = [""]
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=sys_img_file):
        # st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]" + f"\t{prompt}" )
        with st.container():
            with st.spinner("Working on searching answer ..."):
                # st.write(prompt)
                # st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]")
                if prompt:
                    st_tm = time.time()
                    
                    prompt = prompt.lower()
                    if doc_source == "url":
                        context_str = doc_content
                    else:
                        context_str = db_index.get_relevant_documents(prompt)#, k=50, fetch_k=100)
                    stuff_answer = qa_stuff_chain(
                        {
                            "input_documents": context_str, 
                            "question": prompt 
                        }, 
                        return_only_outputs = True
                    )
                    prompt_feedback = stuff_answer["output_text"]
                    
                    print(f"*** chat_llm = {time.time() - st_tm} seconds, *** {prompt_feedback}, ** {stuff_answer['output_text']}")
                    st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]" + "\t" + prompt_feedback)
  
    message = {
        "role": "assistant", 
        "content": prompt, 
        "dt": datetime.now().strftime("%H:%M:%S"),
        "avatar": sys_img_file,
        "feedback": prompt_feedback,
    }
    st.session_state.messages.append(message)

    # print(f"*** st.session_state.messages = {st.session_state.messages}")

    


