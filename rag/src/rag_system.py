from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever 
from langsmith import traceable
from typing import List
from functools import lru_cache
from rag.src.config_schema import RAGConfig
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

class RAGService:
    
    def __init__(self, config_path: str) -> None:
        self.config_path = os.path.abspath(config_path)
        self.config: RAGConfig = self._load_config(self.config_path)

        self.vector_store = None
        self.llm_query = None
        self.llm_generation = None
        self.retriever = None
        self.relevance_chain = None
        self.rag_chain = None
        self.query_rewrite_chain = None

        self._init_rag_system()

    def _load_config(self, config_path: str) -> RAGConfig:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return RAGConfig.model_validate(json.load(f))
        
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

        self.config: RAGConfig = self._load_config(config_path)
        self.config_path = config_path
        self._init_rag_system()

    def _init_rag_system(self) -> None:
        """Initialize the RAG system components based on the current configuration."""
        # Initialize the Chroma Vector Store
        self._load_vector_store()
        # Load model
        self._load_llms_models()
        # Build retriever
        self._build_retriever()
        # Rewrite query chain
        self._build_query_rewrite_chain()
        # Relevance chain to filter documents before generation
        self._build_relevance_chain()
        # Build the main RAG chain
        self._build_rag_chain()
        
    def _load_vector_store(self) -> None:
        self.vector_store = Chroma(
            collection_name=self.config.vector_db.collection_name,
            embedding_function=OpenAIEmbeddings(model=self.config.vector_db.embeddings_model),
            persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        logger.info("Chroma vector store initialized successfully.")    

    def _load_llms_models(self) -> None:
        self.llm_query = ChatOpenAI(model=self.config.models.query_model, temperature=0.0)
        self.llm_generation = ChatOpenAI(model=self.config.models.generation_model, temperature=0.0)
        logger.info("LLM models loaded successfully")
        
    def _build_retriever(self) -> None:
        if self.vector_store is None:
            raise ValueError("Vector store must be initialized before building retriever.")
        if self.llm_query is None:
            raise ValueError("LLM query model must be initialized before building retriever.")
        
        # Retriever MMR (Maximal Marginal Relevance) 
        base_retriever = self.vector_store.as_retriever(
            search_type=self.config.mmr.search_type,
            search_kwargs={
                "k": self.config.mmr.search_k,
                "lambda_mult": self.config.mmr.mmr_diversity_lambda,
                "fetch_k": self.config.mmr.mmr_fetch_k
            }
        )
        # Retriever with cosine similarity
        similarity_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.hybrid_search.search_k}
        )
        # MultiQueryRetriever with MMR
        mmr_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm_query,
            prompt=PromptTemplate.from_template(self.config.prompts.multi_query_prompt)
        )
        # EnsembleRetriever to combine MMR and similarity retrievers
        if self.config.hybrid_search.enable:
            logger.info("Initializing EnsembleRetriever for hybrid search")
            self.retriever = EnsembleRetriever(
                retrievers=[mmr_retriever, similarity_retriever],
                weights=[0.7, 0.3]
            )
        else:
            logger.info("Initializing MMR Retriever without hybrid search")
            self.retriever = mmr_retriever  # Solo MMR

    def _build_query_rewrite_chain(self) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.prompts.rewrite_query_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}"),
        ])
        self.query_rewrite_chain = prompt | self.llm_query | StrOutputParser() # type: ignore

    def _build_relevance_chain(self) -> None:
        relevance_prompt = PromptTemplate.from_template(
            self.config.prompts.relevance_prompt
        )
        self.relevance_chain = relevance_prompt | self.llm_query | StrOutputParser() # type: ignore
        
    def _build_rag_chain(self) -> None:
        # System prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.prompts.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Pregunta: {query}\n\nContexto:\n{context}")
        ])
        self.rag_chain = prompt | self.llm_generation | StrOutputParser()  # type: ignore
        
    @traceable
    def rewrite_query(self, query: str, history) -> str:
        if not history:
            return query
        
        rewritten_query = self.query_rewrite_chain.invoke({ # type: ignore
            "query": query,
            "history": history
        })
        rewritten_query = rewritten_query.strip()
        logger.info(f"Original Query: '{query}'")
        logger.info(f"Rewritten Query: '{rewritten_query}'")
        return rewritten_query or query
        
    @traceable
    def retrieve_documents(self, query: str) -> List:
        """Retrieve relevants documents for a given query."""
        docs = self.retriever.invoke(query) # type: ignore
        return docs
    
    @traceable
    def generate_response(self, query: str, history: list, context: str) -> str:
        """Generate a response base on the query, supported by the context."""
        if self.llm_generation is None:
            raise ValueError("LLM generation model must be initialized before generating response.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.prompts.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Pregunta: {query}\n\nContexto:\n{context}")
        ])

        generation_chain = prompt | self.llm_generation | StrOutputParser()  # type: ignore

        response = generation_chain.invoke({
            "query": query,
            "history": history,
            "context": context,
        })

        return response.strip()

    @traceable
    def filter_relevant_documents(self, docs, query: str) -> list:
        logger.info(f"Filtering {len(docs)} documents for relevance to the query.")
        if self.relevance_chain is None:
            return docs
        filtered_docs = []
        for doc in docs:
            result = self.relevance_chain.invoke({
                "chunk": doc.page_content,
                "query": query
            })

            if result.strip().upper().startswith("SI"):
                filtered_docs.append(doc)
        logger.info(f"{len(filtered_docs)} documents deemed relevant after filtering.")
        # self.documents = filtered_docs
        return filtered_docs

@lru_cache(maxsize=None)
def get_rag_service(config_path: str) -> RAGService:
    normalized_path = os.path.abspath(config_path)
    return RAGService(config_path=normalized_path)    