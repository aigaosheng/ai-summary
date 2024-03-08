#load LLM
import streamlit as st
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
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
from langchain.vectorstores import Chroma, FAISS
import json
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter


"""
Define prompt template
"""
prompt_template_task = {
    "qa": {
        "prompt_template": """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """,
    },
    "summary": {
    }
}
task_id = "qa"
prompt_template2 = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """

prompt_template = PromptTemplate(
            template=prompt_template2, input_variables=["context", "question"]
        )

@st.cache_resource
def llm_chat():
    n_gpu_layers =  -1
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        # callback_manager=callback_manager,
        n_ctx=1024*6, # Uncomment to increase the context window
        # temperature=0.75,
        # f16_kv=True,
        verbose=False,  # Verbose is required to pass to the callback manager
    )
    return llm
    
@st.cache_resource
def llm_embed():
    llama = LlamaCppEmbeddings(model_path="/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf", n_gpu_layers = -1, n_batch = 128)
    return llama
    
@st.cache_data
def get_llm_embed(query, llama_model):
    if isinstance(query, str):
        query = [query]
    doc_result = llama.embed_documents(query)
    return doc_result

@st.cache_data
def pdf_to_pages(file):
	# "extract text (pages) from pdf file"
	# pages = []
	# pdf = pypdf.PdfReader(file)
	# for p in range(len(pdf.pages)):
	# 	page = pdf.pages[p]
	# 	text = page.extract_text()
	# 	pages += [text]
    print(f"*** file = {file}")
    # pdf_loader = PyPDFLoader(file)
    text_splitter = RecursiveCharacterTextSplitter(separators=[".",",","?","!"], chunk_size = 512, chunk_overlap=32, keep_separator = True)

    bytes_data = file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)                      # write data from the uploaded file into it
        pages = PyPDFLoader(tmp.name).load_and_split(text_splitter)      # <---- now it works!
    os.remove(tmp.name)                            # remove temp file
    
    # pages = pdf_loader.load_and_split()
    # print(pages[3].page_content)
    return pages

#load LLM
llm_chat_model = llm_chat()
stuff_chain = load_qa_chain(llm_chat_model, chain_type="stuff", prompt=prompt_template)
llm_embed_model = llm_embed()
#
doc_file = st.file_uploader(label = "PDF Upload", type = ["pdf"], accept_multiple_files = False, help = "upload pdf for chat", disabled = False)

def get_answer(question, context):
    if True:#not (isinstance(context, str) or isinstance(context, list)):
        context_str = context.get_relevant_documents(question)
        # print(f"*** context_cand = {type(context_cand)}")
        # context_str = ""
        # for v in context_cand:
        #     # v = json.loads(v)
        #     # print(f"*** v = {type(v)}, {v}")
        #     context_str += v.page_content + " "
    else:
        context_str = context
        

    print(f"*** context = {len(context_str)}, {context_str}")
        
    stuff_answer = stuff_chain(
        # {"input_documents": context_str, "question": question}, return_only_outputs=True
        {"input_documents": context_str, "question": question}, return_only_outputs=True
    )
    return stuff_answer

@st.cache_resource
def rag_index(doc_file):
    doc_content = pdf_to_pages(doc_file)
    # print(f"*** doc_content = {doc_content[:2]}")
    
    # text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    # docs = text_splitter.split_documents(doc_content)
    # search_type="similarity_score_threshold",
    # search_kwargs={'score_threshold': 0.8}
    vector_index = Chroma.from_documents(doc_content, llm_embed_model).as_retriever(
        search_type="similarity", #"similarity_score_threshold",
        search_kwargs={'k': 16}, #score_threshold': 0.5}
    )

    return vector_index #doc_content #vector_index
    
    
if doc_file:
    vector_index = rag_index(doc_file)
    # print(f"**** {type(doc_content)}")
    ans = get_answer(" how much Segment revenue?", context = vector_index)
    st.write(ans)
