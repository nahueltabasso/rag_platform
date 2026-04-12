from typing import Dict

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever 
import os
from config import *
from rag.src.prompts import MULTI_QUERY_PROMPT, SYSTEM_PROMPT
import streamlit as st

# @st.cache_resource
def initialize_rag_system(config: Dict):
    
    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name=config.get('vector_db', {}).get('collection_name'),
        embedding_function=OpenAIEmbeddings(model=config.get('vector_db', {}).get('embbedings_model')),
        persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db")
    )
    
    # Load model
    llm_query = ChatOpenAI(model=config.get('models', {}).get('query_model'), temperature=0.0)
    llm_generation = ChatOpenAI(model=config.get('models', {}).get('generation_model'), temperature=0.0)
    
    # Retriever MMR (Maximal Marginal Relevance) 
    base_retriever = vector_store.as_retriever(
        search_type=config.get('mmr', {}).get('search_type'),
        search_kwargs={
            "k": config.get('mmr', {}).get('search_k'),
            "lambda_mult": config.get('mmr', {}).get('mmr_diversity_lambda'),
            "fetch_k": config.get('mmr', {}).get('mmr_fetch_k'),
        }
    )
    
    # Retriever with cosine similarity
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.get('hybrid_search', {}).get('search_k')}
    )
    
    # Custom prompt for MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(
        config.get('prompts', {}).get('multi_query_prompt')
    )
    
    # MultiQueryRetriever with MMR
    mmr_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_query,
        prompt=multi_query_prompt
    )
    
    # EnsembleRetriever to combine MMR and similarity retrievers
    if config.get('hybrid_search', {}).get('enable', False):
        retriever = EnsembleRetriever(
            retrievers=[mmr_retriever, similarity_retriever],
            weights=[0.7, 0.3]
        )
    else:
        retriever = mmr_retriever  # Solo MMR
        
    prompt = PromptTemplate.from_template(config.get('prompts', {}).get('system_prompt'))
    
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

def build_relevance_chain(config: Dict, llm_query: ChatOpenAI):
    relevance_prompt = PromptTemplate.from_template(
        config.get('prompts', {}).get('relevance_prompt')
    )
    return relevance_prompt | llm_query | StrOutputParser()

def build_context(query: str, retriever, relevance_chain) -> str:
    docs = retriever.invoke(query)
    relevante_docs = ""
    return ""
    
def filter_relevant_documents(documents, query: str, relevance_chain):
    filtered = []

    for doc in documents:
        result = relevance_chain.invoke({
            "chunk": doc.page_content,
            "query": query
        })

        if result.strip().upper().startswith("SI"):
            filtered.append(doc)

    return filtered    
    
def process_query(config: Dict, query: str):
    try:
        rag_chain, retriever = initialize_rag_system(config)
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