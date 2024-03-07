#load LLM
import streamlit as st
import pypdf
from llama_cpp import Llama

def llm_chat(prompt):
	global llm_inst
	msg = [
		{"role": "system", "content": f"You are an assistant to help summarize the document."},
		{
			"role": "user",
			"content": f"{prompt} "
		}
	]
	try:
		output = llm_inst.create_chat_completion(
			messages = msg,
			max_tokens=200,
			temperature=0.0,
			# stop=[".", "\n"],
		)
	except:
		llm_inst = Llama(
			# model_path="/home/gs/work/llama.cpp/models/llama2-7b/ggml-model-f16.gguf",
			model_path="/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf",
			# model_path="/home/gs/hf_home/models/models--google--gemma-2b/gemma-2b.gguf",
			chat_format="gemma", #"llama-2", #
			# n_gpu_layers=-1, # Uncomment to use GPU acceleration
			# seed=1337, # Uncomment to set a specific seed
			n_ctx=2048, # Uncomment to increase the context window
		)
		print(f"**** LLM load successfully")
		output = llm_inst.create_chat_completion(
			messages = msg,
			max_tokens=200,
			temperature=0.0,
			# stop=[".", "\n"],
		)
	result = output["choices"][0]["message"]["content"]
	return result

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
		output = llm_inst.create_chat_completion(
			messages = msg,
			max_tokens=200,
			temperature=0.0,
			# stop=[".", "\n"],
		)
	except:
		llm_inst = Llama(
			# model_path="/home/gs/work/llama.cpp/models/llama2-7b/ggml-model-f16.gguf",
			model_path="/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf",
			# model_path="/home/gs/hf_home/models/models--google--gemma-2b/gemma-2b.gguf",
			chat_format="gemma", #"llama-2", #
			# n_gpu_layers=-1, # Uncomment to use GPU acceleration
			# seed=1337, # Uncomment to set a specific seed
			n_ctx=2048, # Uncomment to increase the context window
		)
		print(f"**** LLM load successfully")
		output = llm_inst.create_chat_completion(
			messages = msg,
			max_tokens=200,
			temperature=0.0,
			# stop=[".", "\n"],
		)
	result = output["choices"][0]["message"]["content"]
	return result


def pdf_to_pages(file):
	"extract text (pages) from pdf file"
	pages = []
	pdf = pypdf.PdfReader(file)
	for p in range(len(pdf.pages)):
		page = pdf.pages[p]
		text = page.extract_text()
		pages += [text]
	return pages

doc_file = st.file_uploader(label = "PDF Upload", type = ["pdf"], accept_multiple_files = False, help = "upload pdf for chat", disabled = False)

if doc_file:
    doc_content = pdf_to_pages(doc_file)
    print(f"**** {type(doc_content)}")
    st.write(" ****** \n".join(doc_content))
