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

#Title
st.set_page_config(page_title=" Jack of All Trades - 百晓通")

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




# Hugging Face Credentials
# Store store history
user_img_file = "./resource/logo.jpeg"
sys_img_file = "./resource/logo.jpeg"

with st.sidebar:
    st.image(sys_img_file, caption = "Avatar learning")
    st.markdown(' enjoy fun leaning adventure')

#
doc_file = st.file_uploader(
    label = "PDF Upload", 
    type = ["pdf"], 
    accept_multiple_files = False, 
    help = "upload pdf for summary & ask me anything in the document", 
    disabled = False
)

if doc_file:
    doc_content = pdf_processor.load_pdf(doc_file)
    # print(f"***111** doc_content = {llm_chat_model} {doc_content}")
    ans = summ_stuff_chain.run(doc_content[:3])
    st.write(ans)

# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [
#         {
#             "role": "assistant", 
#             "content": "Welcome to leaning course", 
#             "audio": None,
#             "video": None,
#             "dt": datetime.now().strftime("%H:%M:%S"),
#             "avatar": sys_img_file,
#             "feedback": "",
#         }
#     ]

# # Display chat messages
# for message in st.session_state.messages:
#     print(message)
#     with st.chat_message(message["role"], avatar = message["avatar"]):
#         if message["role"] == "assistant":
#             if message["feedback"] is not None:
#                 st.write(":orange[" + message["dt"] + "]\t" + message["feedback"])
#             else:
#                 st.write(":orange[" + message["dt"] + "]\t")
#         else:
#             if message["content"] is not None:
#                 st.write(":orange[" + message["dt"] + "]\t" + message["content"])
#             else:
#                 st.write(":orange[" + message["dt"] + "]\t")
#         # if message["audio"]:
#         #     # st.audio(message["audio"])
#         #     st.markdown(
#         #         autoplay_audio(message["audio"], False),
#         #         unsafe_allow_html=True,
#         #     )

#         if message["video"]:
#             # st.video(message["video"])
#             st.markdown(
#                 autoplay_video(message["video"], False),
#                 unsafe_allow_html=True,
#             )
            
        
# #chat counter
# if "wav_counter" not in  st.session_state:
#     st.session_state["wav_counter"] = 0
# else:
#     st.session_state["wav_counter"] += 1

# # User-provided prompt
# if prompt := st.chat_input():
#     st.session_state.messages.append(
#         {
#             "role": "user", 
#             "content": prompt, 
#             "audio": None, 
#             "video": None,
#             "dt": datetime.now().strftime("%H:%M:%S"),
#             "avatar": user_img_file,
#             "feedback": "",
#         }
#     )
#     with st.chat_message("user", avatar=user_img_file):
#         st.write(":orange[" + datetime.now().strftime("%H:%M:%S") + "]\t" + prompt)

# # TTS response
# prompt_feedback = [""]
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant", avatar=sys_img_file):
#         # st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]" + f"\t{prompt}" )
#         with st.container():
#             with st.spinner("Working on voice generation ..."):
#                 # st.write(prompt)
#                 # st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]")

#                 #call chat LLM
#                 prompt_msg = [
#                       {
#                           "role": "assistant", 
#                            "content": "You are a teach assistant"
#                       },
#                       {
#                           "role": "user",
#                           "content": prompt,
#                       }
#                   ]
#                 st = time.time()
#                 prompt_feedback = chat_llm(prompt_msg)

#                 prompt_feedback2 = "\n".join(prompt_feedback)
#                 print(f"*** chat_llm = {time.time() - st} seconds")
#                 st.write(":orange[" + str(datetime.now().strftime("%H:%M:%S")) + "]" + "\t" + prompt_feedback2)
                
#                 #
#                 # tmp_wav_file_stack = []
#                 # video_pth_stack = []
#                 # for k, pfk in enumerate(["\n".join(prompt_feedback)]):
#                 pfk = prompt_feedback2
#                 k = 0
#                 tmp_wav_file = f"./tmp/tmpwav{st.session_state['wav_counter']}_{k}.wav"
#                 print(f"**** {k}, {pfk}, {tmp_wav_file}")

#                 st = time.time()
#                 tts(pfk, lang = "eng", default_file = tmp_wav_file) 
#                 print(f"*** tts = {time.time() - st} seconds")

#                 st = time.time()
#                 video_pth = voice2video("./img/cartoon-640-360.jpg", tmp_wav_file)
#                 # st.video(video_pth, format="video/mp4")#"./tmp/gs-avatar.mp4")
#                 print(f"*** video = {time.time() - st} seconds")
                
#                 st.markdown(
#                     autoplay_video(video_pth),
#                     unsafe_allow_html=True,
#                 )
#                 # tmp_wav_file_stack.append(tmp_wav_file)
#                 # video_pth_stack.append(video_pth)
                    


#     message = {
#         "role": "assistant", 
#         "content": prompt, 
#         "audio": tmp_wav_file,  
#         "video": video_pth,
#         "dt": datetime.now().strftime("%H:%M:%S"),
#         "avatar": sys_img_file,
#         "feedback": "\n".join(prompt_feedback),
#     }
#     st.session_state.messages.append(message)

#     print(f"*** st.session_state.messages = {st.session_state.messages}")

    


