from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from config import *
from rag.src.prompts import MULTI_QUERY_PROMPT, SYSTEM_PROMPT
import streamlit as st

# @st.cache_resource
def initialize_rag_system():
    
    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name='site_wwII_wikipedia',
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=CHROMA_DB_PATH
    )
    
    # Load model
    llm_query = ChatOpenAI(model=QUERY_MODEL, temperature=0.0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0.0)
    
    # Retriever MMR (Maximal Marginal Relevance) 
    base_retriever = vector_store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K,
        }
    )
    
    # Custom prompt for MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)
    
    # MultiQueryRetriever with MMR
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_query,
        prompt=multi_query_prompt
    )
    
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    
    rag_chain = (
        {
            "context": retriever | format_documents,
            "query": RunnablePassthrough()
        } 
        | prompt 
        | llm_generation 
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def format_documents(documents):
    print("INIT FORMAT DOCUMENTS ----")
    formatted = []
    for i, doc in enumerate(documents):
        formatted.append(f"Fragmento {i+1}:\n{doc.metadata}\n{doc.page_content}\n")
    
    for f in formatted:
        print(f"Formatted document:\n{f}\n{'-'*50}")
    print("END FORMAT DOCUMENTS ----")
    return "\n\n".join(formatted)

def process_query(query: str):
    try:
        rag_chain, retriever = initialize_rag_system()
        
        # Obtain response
        response = rag_chain.invoke(query)
        
        # Obtain relevant documents
        docs = retriever.invoke(query)
        
        # Format documents for display
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "chunk": i,
                "content": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "url": doc.metadata.get("source", 'N/A')
            }
            docs_info.append(doc_info)
        return response, docs_info
    except Exception as e:
        error_msg = f"Error al procesar la consulta: {str(e)}"
        return error_msg, []
    
def get_retriever_info():
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + Multiquery",
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": 'N/A'
    }