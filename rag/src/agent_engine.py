from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing import Dict, List
from rag.src.config import DATA_DIR
from rag.src.config_schema import RAGConfig
from rag.src.rag_system import RAGService
from rag.src.state import AppState, RouteDecisionDTO
from rag.src.utils import format_documents
import sqlite3
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

class MemoryRAGAgentEngine:
    
    def __init__(self, user_id: str, rag_service: RAGService) -> None:
        self.user_id = user_id
        self.rag_service = rag_service
        self.config_app: RAGConfig = rag_service.config
        self.checkpointer = None
        self.llm = ChatOpenAI(model=self.config_app.models.generation_model, temperature=0.2)
        self.router_llm = ChatOpenAI(model=self.config_app.models.router_llm, temperature=0.0)
        self.router_chain = None
        self.chitchat_chain = None
        
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
        self._build_router_chain()
        self._build_chitchat_chain()
        self.workflow = self._build_workflow()
        
    def _build_router_chain(self) -> None:
        """Build the router chain to use in the router node."""
        logger.info("Building router chain for MemoryRAGAgentEngine")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config_app.prompts.router_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])    
        
        structured_router = self.router_llm.with_structured_output(RouteDecisionDTO)
        self.router_chain = prompt | structured_router
        
    def _build_chitchat_chain(self) -> None:
        """Build the chitchat chain to use in the chitchat node."""
        logger.info("Building chitchat chain for MemoryRAGAgentEngine")
        topic = self.config_app.topic
        prompt = self.config_app.prompts.chitchat_prompt
        chitchat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt.format(topic=topic)),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        self.chitchat_chain = chitchat_prompt | self.llm | StrOutputParser()  # type: ignore

        
    def _init_checkpointer(self) -> None:
        """Initialize a global checkpointer for the system."""  
        logger.info("Initializing checkpointer for MemoryRAGAgentEngine")
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(
            os.path.join(DATA_DIR, "memory_checkpointer.db"),
            check_same_thread=False,
        )
        self.checkpointer = SqliteSaver(conn)
        
    def _build_workflow(self):
        """Build a LangGraph workflow that integrates the RAG chain with memory
        management."""
        logger.info("Building workflow for MemoryRAGAgentEngine")
        workflow = StateGraph(state_schema=AppState)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("route_decision", self.router_node)
        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("filter_relevant_documents", self.filter_relevant_documents_node)
        workflow.add_node("format_context", self.format_context_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("generate_chitchat_response", self.generate_chitchat_response_node)
        workflow.add_node("user_memory_retrieval", self.user_memory_retrieval_node)
        
        workflow.add_edge(START, "route_decision")
        workflow.add_conditional_edges("route_decision", 
                                       lambda state: state["route_decision"],
                                       {
                                            # Mapp: "Pydantic Value" - "Next node"
                                            "EXPERT_RAG": "rewrite_query",
                                            "USER_MEMORY": "user_memory_retrieval",
                                            "CHITCHAT": "generate_chitchat_response"
                                       })
        # RAG Expert workflow
        workflow.add_edge("rewrite_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "filter_relevant_documents")
        workflow.add_edge("filter_relevant_documents", "format_context")
        workflow.add_edge("format_context", "generate_response")    
        workflow.add_edge("generate_response", END)
        
        # Memory User workflow
        workflow.add_edge("user_memory_retrieval", END)
        
        # Chitchat workflow
        workflow.add_edge("generate_chitchat_response", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def generate_chitchat_response_node(self, state: AppState) -> Dict:
        """Node to generate a response for chitchat queries."""
        logger.info("Enter to generate_chitchat_response_node()")
        query = state["query"]
        history = state["messages"] 
        history = self._get_history_chat(history=history, query=query)
        response = self.chitchat_chain.invoke({"query": query, "history": history}) # type: ignore
        
        return {"messages": [AIMessage(content=response)]}
    
    def user_memory_retrieval_node(self, state: AppState) -> Dict:
        """Node to retrieve user memory."""
        logger.info("Enter to user_memory_retrieval_node()")
        return {"messages": [AIMessage(content="Hola! Este es un mensaje predeterminado del user memory retrieval node")]}
        
    def router_node(self, state: AppState) -> Dict:
        """Node to route que user input to a specific workflow path based on the content of the message."""
        logger.info("Enter to router_node()")
        query = state["query"]
        history = state["messages"]
        
        history = self._get_history_chat(history=history, query=query)
        if query is None or query.strip() == "":
            raise ValueError("Query can not be empty.")
        
        result: RouteDecisionDTO = self.router_chain.invoke( # type: ignore
            {"query": query, "history": history}
        ) # type: ignore
        
        logger.info(f"🚦 [Router Decision] - {result.route} - for query: '{query}'")
        
        return {"route_decision": result.route}
        
    def rewrite_query_node(self, state: AppState) -> Dict:
        """Node to rewrite the user query based on the conversation history."""
        logger.info("Enter to rewrite_query_node()")
        query = state["query"]
        history = state["messages"]
        history = self._get_history_chat(history=history, query=query)
        rewritten_query = self.rag_service.rewrite_query(query=query,
                                                         history=history)
        return {"rewritten_query": rewritten_query}
        
    def retrieve_documents_node(self, state: AppState) -> Dict:
        """Node to retrieve documents based on the rewritten query."""
        logger.info("Enter to retrieve_documents_node()")
        query = state["rewritten_query"]
        
        if not query or query.strip() == "":
            raise ValueError("Query can not be empty.")
        
        docs = self.rag_service.retrieve_documents(query=query)        
        return {"context_docs": docs}
        
    def filter_relevant_documents_node(self, state: AppState) -> Dict:
        """Node to filter the retrieved documents based on their relevance to the query."""
        logger.info("Enter to filter_relevant_documents_node()")
        docs = state["context_docs"]
        query = state["rewritten_query"]
        
        if not docs:
            return {"context_docs": []}
        if not query or query.strip() == "":
            raise ValueError("Query can not be empty.")
        
        relevants_docs = self.rag_service.filter_relevant_documents(docs=docs, query=query)
        
        return {"context_docs": relevants_docs}
    
    def format_context_node(self, state: AppState) -> Dict:
        """Node to format a context."""
        logger.info("Enter to format_context_node()")
        docs = state["context_docs"]
        if not docs:
            return {"formatted_context": "No se encontraron documentos relevantes."}
        context = format_documents(documents=docs)
        return {"formatted_context": context}
    
    def generate_response_node(self, state: AppState) -> Dict:
        """Node to generate a response to a user query based on the context provided by the retrieved documents."""
        logger.info("Enter to generate_response_node()")
        query = state["rewritten_query"]
        messages = state["messages"]
        context = state["formatted_context"]

        history = self._get_history_chat(history=messages, query=query)
        response = self.rag_service.generate_response(
            query=query,
            history=history,
            context=context
        )
        logger.info(f"[RAG_EXPERT] Response -> '{response}' for query: '{query}'")
        return {"messages": [AIMessage(content=response)]}
    
    @traceable
    def chat(self, message: str, thread_id: str="default"):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            logger.info(f"Chat invoked with message: '{message}' and thread_id: '{thread_id}'")
            result = self.workflow.invoke(
                {"messages": [HumanMessage(content=message)], "query": message}, config # type: ignore
            )
            
            assistant_response = result["messages"][-1].content
            return assistant_response
        except Exception as e:
            logger.error(f"Error processing the message: {str(e)}", exc_info=True)
            return f"Error processing the message: {str(e)}"
    
    def _get_history_chat(self, history: List, query: str) -> List:
        if history and isinstance(history[-1], HumanMessage):
            last_message = history[-1]
            if last_message.content == query:
                history = history[:-1]
                
        history = self.message_trimmer.invoke(history) # type: ignore
        return history
        
        
        
        