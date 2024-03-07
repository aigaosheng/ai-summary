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
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile

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

prompt_template = PromptTemplate(
            template=prompt_template_task[task_id]["prompt_template"], input_variables=["context", "question"]
        )


def llm_chat():
    n_gpu_layers =  -1
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        # callback_manager=callback_manager,
        n_ctx=2048, # Uncomment to increase the context window
        # temperature=0.75,
        # f16_kv=True,
        verbose=False,  # Verbose is required to pass to the callback manager
    )
    return llm

def llm_embed(prompt):
	global llm_embed_inst
	msg = [
		{"role": "system", "content": f"You are an assistant to help summarize the document."},
		{
			"role": "user",
			"content": f"{prompt} "
		}
	]
	try:
		output = llm_embed_inst(prompt)
	except:
		llm_embed_inst = Llama(
			# model_path="/home/gs/work/llama.cpp/models/llama2-7b/ggml-model-f16.gguf",
			model_path="/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf",
			# model_path="/home/gs/hf_home/models/models--google--gemma-2b/gemma-2b.gguf",
			chat_format="gemma", #"llama-2", #
			# n_gpu_layers=-1, # Uncomment to use GPU acceleration
			# seed=1337, # Uncomment to set a specific seed
			n_ctx=2048, # Uncomment to increase the context window
            embeddings = True,
		)
		print(f"**** LLM load successfully")
		output = llm_embed_inst(prompt)
	result = output["choices"][0]["message"]["content"]
	return result


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

    bytes_data = file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)                      # write data from the uploaded file into it
        pages = PyPDFLoader(tmp.name).load_and_split()      # <---- now it works!
    os.remove(tmp.name)                            # remove temp file

    # pages = pdf_loader.load_and_split()
    # print(pages[3].page_content)
    return pages

#load LLM
llm_chat_model = llm_chat()
stuff_chain = load_qa_chain(llm_chat_model, chain_type="stuff", prompt=prompt_template)
#
doc_file = st.file_uploader(label = "PDF Upload", type = ["pdf"], accept_multiple_files = False, help = "upload pdf for chat", disabled = False)

def get_answer(question, context):
    stuff_answer = stuff_chain(
        {"input_documents": context, "question": question}, return_only_outputs=True
    )
    return stuff_answer

if doc_file:
    doc_content = pdf_to_pages(doc_file)
    # print(f"**** {type(doc_content)}")
    ans = get_answer("how many revenue in 2023?", doc_content[:2])
    st.write(ans)
