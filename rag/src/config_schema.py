from pydantic import BaseModel, Field
from typing import Literal

class VectorDBConfig(BaseModel):

    type: Literal['ChromaDB']
    collection_name: str
    embeddings_model: str
    chunk_size: int
    chunk_overlap: int
    
class ModelConfig(BaseModel):
    
    provider: Literal['OpenAI']
    query_model: str
    generation_model: str
    
class MMRRetrieverConfig(BaseModel):
    
    search_type: str = 'mmr'
    mmr_diversity_lambda: float = 0.7
    mmr_fetch_k: int = 20
    search_k: int = 5
    
class HybridSearchConfig(BaseModel):
    
    enable: bool = False
    similarity_threshold: float = 0.75
    search_k: int = 5
    
class PromptConfig(BaseModel):
    
    system_prompt: str
    multi_query_prompt: str
    relevance_prompt: str
    rewrite_query_prompt: str
    
class MemoryStateConfig(BaseModel):
    
    max_messages: int = 5
    
class RAGConfig(BaseModel):
    
    name: str
    topic: str
    page_icon: str
    initial_message: str
    input_field_legend: str
    footer_message: str
    vector_db: VectorDBConfig
    models: ModelConfig
    mmr: MMRRetrieverConfig
    hybrid_search: HybridSearchConfig
    prompts: PromptConfig
    memory: MemoryStateConfig