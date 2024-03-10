import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate

model_name = "/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf"
model_name_embed = "/home/gs/hf_home/models/models--google--gemma-2b/gemma-2b.gguf"

"""
Define prompt template for QA
"""
qa_prompt_template_cfg = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """
qa_prompt_template = PromptTemplate(
    template = qa_prompt_template_cfg, 
    input_variables = ["context", "question"]
)

"""
define prompt template for summary
"""
sum_prompt_template_cfg = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
"""
sum_prompt_template = PromptTemplate(
    template = sum_prompt_template_cfg, 
    input_variables=["text"]
)

# @st.cache_resource
# def get_summary(docs, _llm_engine):
#     # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#     stuff_chain = load_summarize_chain(_llm_engine, chain_type = "stuff", prompt = sum_prompt_template)
#     output = stuff_chain.run(docs)

#     return output

#define consistent parametes
# n_batch >= chunk-size
chunk_size = 512
@st.cache_data
def load_embed_model():
    llm_embed_model = LlamaCppEmbeddings(
        model_path = model_name_embed, 
        n_gpu_layers = -1, 
        n_ctx = 512 * 4, 
        n_batch = chunk_size, 
        verbose=False
    )

    return llm_embed_model

@st.cache_data
def load_chat_model():
    llm_chat_model = LlamaCpp(
            model_path = model_name,
            n_gpu_layers = -1,
            n_batch = chunk_size,
            # callback_manager=callback_manager,
            n_ctx=1024*4, # Uncomment to increase the context window
            # temperature=0.75,
            # f16_kv=True,
            verbose=False,  # Verbose is required to pass to the callback manager
    )

    return llm_chat_model

    
