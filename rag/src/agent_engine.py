from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, trim_messages
from rag.src.config import DATA_DIR
from rag.src.rag_system import RAGService
import sqlite3
import os

from rag.src.state import AppState
from rag.src.utils import format_documents
from typing import Dict

class MemoryRAGAgentEngine:
    
    def __init__(self, user_id: str, rag_service: RAGService) -> None:
        self.user_id = user_id
        self.rag_service = rag_service
        self.checkpointer = None
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        
        # Message trimming config
        self.message_trimmer = trim_messages(
            strategy="last",
            max_tokens=4000,
            token_counter=self.llm,
            start_on="human",
            include_system=True
        )
        # Set a global checkpointer for the system
        self._init_checkpointer()
        
        
    def _init_checkpointer(self) -> None:
        """Initialize a global checkpointer for the system."""  
        conn = sqlite3.connect(os.path.join(DATA_DIR, "memory_checkpointer.db"))
        self.checkpointer = SqliteSaver(conn)
        
        
    def _build_workflow(self):
        """Build a LangGraph workflow that integrates the RAG chain with memory
        management."""
        
        workflow = StateGraph(state_schema=AppState)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("filter_relevant_documents", self.filter_relevant_documents_node)
        workflow.add_node("format_context", self.format_context_node)
        workflow.add_node("generate_response", self.generate_response_node)
        
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "filter_relevant_documents")
        workflow.add_edge("filter_relevant_documents", "format_context")
        workflow.add_edge("format_context", "generate_response")    
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
        
        
    def rewrite_query_node(self, state: AppState) -> Dict:
        query = state["query"]
        history = state["messages"]
        if history and isinstance(history[-1], HumanMessage):
            last_message = history[-1]
            if last_message.content == query:
                history = history[:-1]
                
        history = self.message_trimmer.invoke(history) # type: ignore
        rewritten_query = self.rag_service.rewrite_query(query=query,
                                                         history=history)
        return {"rewritten_query": rewritten_query}
        
    def retrieve_documents_node(self, state: AppState) -> Dict:
        query = state["rewritten_query"]
        
        if not query or query.strip() == "":
            raise ValueError("Query can not be empty.")
        
        docs = self.rag_service.retrieve_documents(query=query)        
        return {"context_docs": docs}
        
    def filter_relevant_documents_node(self, state: AppState) -> Dict:
        docs = state["context_docs"]
        query = state["rewritten_query"]
        
        if not docs:
            return {"context_docs": []}
        if not query or query.strip() == "":
            raise ValueError("Query can not be empty.")
        
        relevants_docs = self.rag_service.filter_relevant_documents(docs=docs, query=query)
        
        return {"context_docs": relevants_docs}
    
    def format_context_node(self, state: AppState) -> Dict:
        docs = state["context_docs"]
        if not docs:
            return {"formatted_context": "No se encontraron documentos relevantes."}
        context = format_documents(documents=docs)
        return {"formatted_context": context}
    
    def generate_response_node(self, state: AppState) -> Dict:
        query = state["rewritten_query"]
        messages = state["messages"]
        context = state["formatted_context"]

        history = messages
        if history and isinstance(history[-1], HumanMessage):
            if history[-1].content == query:
                history = history[:-1]
        history = self.message_trimmer.invoke(history) # type: ignore
        response = self.rag_service.generate_response(
            query=query,
            history=history,
            context=context
        )
        return {"messages": [AIMessage(content=response)]}
    