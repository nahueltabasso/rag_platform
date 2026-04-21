from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever 
from operator import itemgetter
from utils import format_documents
import os
import json

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

class RAGService:
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str) -> None:
        config_path = str(os.path.abspath(config_path))
        if self.__class__._initialized:
            if getattr(self, "config_path", None) != config_path:
                self.set_config(config_path)
            return

        self.config = self._load_config(config_path)
        self.config_path = config_path
        
        self.vector_store = None
        self.llm_query = None
        self.llm_generation = None
        self.retriever = None
        self.rag_chain = None
        self.rag_chain_with_memory = None
        self.query_rewrite_chain = None
        self.documents = []
        self.store = {}
        
        self._init_rag_system()
        self.__class__._initialized = True
        
    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def set_config(self, config_path: str) -> None:
        """Set RAG system configuration

        Args:
            config_path (str): Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        logger.info(f"Setting new configuration for RAG system: {config_path}")
        config_path = str(os.path.abspath(config_path))
        if getattr(self, "config_path", None) == config_path:
            return

        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.documents = []
        self._init_rag_system()
        
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()    
            
        history = self.store[session_id]
        max_messages = self.config.get('memory', {}).get('max_messages', 10)
        if len(history.messages) > max_messages:
            history.messages = history.messages[-max_messages:]
        
        return history
        
    def _init_rag_system(self) -> None:
        """Initialize the RAG system components based on the current configuration."""
        
        # Initialize the Chroma Vector Store
        self.vector_store = Chroma(
            collection_name=self.config.get('vector_db', {}).get('collection_name'),
            embedding_function=OpenAIEmbeddings(model=self.config.get('vector_db', {}).get('embbedings_model')),
            persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        logger.info("Chroma vector store initialized successfully.")
        
        # Load model
        self.llm_query = ChatOpenAI(model=self.config.get('models', {}).get('query_model'), temperature=0.0)
        self.llm_generation = ChatOpenAI(model=self.config.get('models', {}).get('generation_model'), temperature=0.0)
        logger.info("LLM models loaded successfully")

        # Retriever MMR (Maximal Marginal Relevance) 
        base_retriever = self.vector_store.as_retriever(
            search_type=self.config.get('mmr', {}).get('search_type'),
            search_kwargs={
                "k": self.config.get('mmr', {}).get('search_k'),
                "lambda_mult": self.config.get('mmr', {}).get('mmr_diversity_lambda'),
                "fetch_k": self.config.get('mmr', {}).get('mmr_fetch_k'),
            }
        )
        
        # Retriever with cosine similarity
        similarity_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.get('hybrid_search', {}).get('search_k')}
        )

        # Custom prompt for MultiQueryRetriever
        multi_query_prompt = PromptTemplate.from_template(
            self.config.get('prompts', {}).get('multi_query_prompt')
        )
        
        # MultiQueryRetriever with MMR
        mmr_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm_query,
            prompt=multi_query_prompt
        )
    
        # EnsembleRetriever to combine MMR and similarity retrievers
        if self.config.get('hybrid_search', {}).get('enable', False):
            logger.info("Initializing EnsembleRetriever for hybrid search")
            self.retriever = EnsembleRetriever(
                retrievers=[mmr_retriever, similarity_retriever],
                weights=[0.7, 0.3]
            )
        else:
            logger.info("Initializing MMR Retriever without hybrid search")
            self.retriever = mmr_retriever  # Solo MMR
        
        # System prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.get('prompts', {}).get('system_prompt')),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Pregunta: {query}\n\nContexto:\n{context}")
        ])
        
        # Rewrite query chain
        self.query_rewrite_chain = self._build_query_rewrite_chain()
        
        # Relevance chain to filter documents before generation
        relevance_chain = self._build_relevance_chain()

        self.rag_chain = (
            {
                "query": itemgetter("query"),
                "history": itemgetter("history"),
                # "context": lambda query: self._build_context(query=query, relevance_chain=relevance_chain),
                "context": lambda x: self._build_context(
                    query=x["query"],
                    history=x["history"],
                    relevance_chain=relevance_chain,
                ),
            } 
            | prompt 
            | self.llm_generation 
            | StrOutputParser()
        )
        
        self.rag_chain_with_memory = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="query", 
            history_messages_key="history"
        )

    def process_query(self, query: str, session_id: str = "default") -> tuple:
        """Process a user query through the RAG system and return the response along with relevant documents.
        
        Args:            
            query (str): The user query to process.
            
        Returns:
            tuple: A tuple containing the response string and a list of relevant documents information.
        """
        logger.info(f"Processing query: {query}")
        try:
            # Obtain response
            # response = self.rag_chain.invoke(query) # type: ignore
            response = self.rag_chain_with_memory.invoke( # type: ignore
                {"query": query},
                config={"configurable": {"session_id": session_id}}
            )
            # Obtain relevant documents
            docs = self.documents
            # Format documents for display
            docs_info = []
            for i, doc in enumerate(docs[:self.config.get('hybrid_search', {}).get('search_k', 5)], 1):
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

    def _build_query_rewrite_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.get('prompts', {}).get('rewrite_query_prompt')),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}"),
        ])
        
        return prompt | self.llm_query | StrOutputParser() # type: ignore
    
    def _rewrite_query(self, query: str, history) -> str:
        if not history:
            return query
        
        rewritten_query = self.query_rewrite_chain.invoke({ # type: ignore
            "query": query,
            "history": history
        })
        rewritten_query = rewritten_query.strip()
        logger.info(f"Original Query: '{query}")
        logger.info(f"Rewritten Query: '{rewritten_query}'")
        return rewritten_query or query
        

    def _build_relevance_chain(self):
        relevance_prompt = PromptTemplate.from_template(
            self.config.get('prompts', {}).get('relevance_prompt')
        )
        return relevance_prompt | self.llm_query | StrOutputParser() # type: ignore
    
    def _build_context(self, query: str, history: list, relevance_chain) -> str:
        rewritten_query = self._rewrite_query(query=query, history=history)
        docs = self.retriever.invoke(rewritten_query) # type: ignore
        relevante_docs = self._filter_relevant_documents(docs=docs,
                                                         query=query,
                                                         relevance_chain=relevance_chain)
        return format_documents(relevante_docs)
    
    def _filter_relevant_documents(self, docs, query: str, relevance_chain):
        logger.info(f"Filtering {len(docs)} documents for relevance to the query.")
        filtered_docs = []
        for doc in docs:
            result = relevance_chain.invoke({
                "chunk": doc.page_content,
                "query": query
            })

            if result.strip().upper().startswith("SI"):
                filtered_docs.append(doc)
        logger.info(f"{len(filtered_docs)} documents deemed relevant after filtering.")
        self.documents = filtered_docs
        return filtered_docs

def get_rag_service(config_path: str) -> RAGService:
    return RAGService(config_path=config_path) # type: ignore
    