from pypdf import PdfReader
from copy import deepcopy
from langchain_core.documents.base import Document

import streamlit as st

def paragraph_process(doc_pages, source = None, company = None, year = None):
    def single_page(i_text, page_id, is_lower = True):
        i_text = i_text.lower() if is_lower else i_text
        
        nn = len(i_text)
        pg_stack = [] #paragraph list
        ss = ""
        k = 0
        while k < nn:
            if i_text[k] == "\n":
                kp = k + 1
                rt_ct = 0
                while kp < nn:
                    if i_text[kp] == "\n":
                        rt_ct += 1
                        kp += 1
                    elif i_text[kp] == " ":
                        kp += 1
                        continue
                    else:
                        break
                if rt_ct >= 1:
                    #paragraph
                    # ss += "\n"
                    pg_stack.append(ss)
                    ss = ""
                    k = kp
                else:
                    ss += " "
                    k += 1
            elif i_text[k] == " ":
                #space
                kp = k + 1
                rt_ct = 0
                while kp < nn:
                    if i_text[kp] == " ":
                        rt_ct += 1
                        kp += 1
                    else:
                        break
                ss += " "
                k = kp            
            else:
                ss += i_text[k]
                k += 1
        # pg = list(map(lambda x: x.strip(), filter(lambda x: len(x) >= 1, ss.split("\n"))))
        try:
            pg_id = int(pg_stack[-1])
            pg_stack = pg_stack[:-1] if pg_id - 1 == page_id else pg_stack
        except:
            try:
                pg_id = int(pg_stack[0])
                pg_stack = pg_stack[1:] if pg_id - 1 == page_id else pg_stack
            except:
                pg_id = None
        
        # print(f"{page_id}, {pg_id}, {pg_stack}")
        # print(f"** {ss}, {k}")
        return pg_stack

    #process page-by-page
    para_docs = []
    para_doc_pid = []
    for pid, pcontent in enumerate(doc_pages):
        ss = single_page(pcontent, pid)
        para_docs.extend(ss)
        para_doc_pid.extend([{"page": pid, "source": source, "company": company, "year": 2023}] * len(ss))

    return para_docs, para_doc_pid

def chunk_create(content_lst: list[str], metadata_lst: list[{}] = None, chunk_size: int = 256):
    k = 0
    chunk_stack = []
    chunk_meta_stack = []
    chunk_str = ""
    nn = len(content_lst)
    metadata_lst = metadata_lst if metadata_lst else [{}] * nn
    doc_ct = 0
    while k < nn:
        if len(chunk_str.split(" ")) >= chunk_size:
            chunk_meta = metadata_lst[k - 1]
            chunk_meta["docid"] = deepcopy(doc_ct)              
            tmpdoc = Document(page_content = chunk_str.strip(), metadata=chunk_meta)
            chunk_stack.append(tmpdoc)
            chunk_str = content_lst[k]
            doc_ct += 1
        else:
            tmpstr = chunk_str + " " + content_lst[k]
            if len(tmpstr.split(" ")) > chunk_size and len(content_lst[k].split(" ")) >= chunk_size//2:
                chunk_meta = metadata_lst[k - 1]
                chunk_meta["docid"] = deepcopy(doc_ct)                
                tmpdoc = Document(page_content = chunk_str.strip(), metadata=chunk_meta)
                chunk_stack.append(tmpdoc)
                # chunk_stack.append(chunk_str.strip())
                chunk_str = content_lst[k]
                doc_ct += 1
            else:
                chunk_str = tmpstr
        # print(f"** {k}, {chunk_str}, {content_lst[k]}")
        k += 1
        
    if len(chunk_str.split(" ")) > 0 and len(chunk_str.split(" ")) < chunk_size//2:
        if len(chunk_stack): 
            tmpstr = chunk_stack[-1].page_content + " " + chunk_str
            chunk_meta = metadata_lst[nn - 1]
            chunk_meta["docid"] = deepcopy(doc_ct) 
            chunk_stack[-1] = Document(page_content = tmpstr.strip(), metadata=chunk_meta)
        else:
            chunk_meta = metadata_lst[nn - 1]
            chunk_meta["docid"] = deepcopy(doc_ct) + 1
            tmpdoc = Document(page_content = tmpstr.strip(), metadata=chunk_meta)
            chunk_stack.append(tmpdoc)
            # chunk_stack.append(chunk_str.strip())
    else:
        chunk_meta = metadata_lst[nn - 1]
        chunk_meta["docid"] = deepcopy(doc_ct) + 1
        tmpdoc = Document(page_content = tmpstr.strip(), metadata=chunk_meta)
        chunk_stack.append(tmpdoc)
        # chunk_stack.append(chunk_str.strip())

    # print(chunk_stack)
    return chunk_stack
    
@st.cache_data       
def load_pdf(pfile, chunk_size = 256):
    # print(f"*** {pfile}, doc_pages = ")
    reader = PdfReader(pfile)
    # print(f"*** {reader}, doc_pages ")
    doc_pages = list(map(lambda x: x.extract_text(extraction_mode="layout"), reader.pages))
    # print(f"*** {pfile}, doc_pages = {doc_pages}")
    para_docs, para_meta = paragraph_process(doc_pages, source = pfile, company = "", year = "")

    #
    chunk_docs = chunk_create(para_docs, para_meta, chunk_size = chunk_size)
    
    return chunk_docs

@st.cache_data       
def chunk_doc(para_docs, para_meta, chunk_size = 256):
    chunk_docs = chunk_create(para_docs, para_meta, chunk_size = chunk_size)

    return chunk_docs