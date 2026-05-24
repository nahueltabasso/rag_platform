from typing import List, Dict, Any, Literal, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Extended State which combine messages with vector memory
class AppState(TypedDict):
    """State that combines LangGraph messages with vector memory."""
    
    messages: Annotated[List[BaseMessage], add_messages]
    vector_memories: List[str] # Vector memories actives IDs
    user_profile: Dict[str, Any] # User profile information
    last_memory_extraction: Optional[str] # Last processed message for memory extraction
    route_decision: str

    query: str
    rewritten_query: str
    context_docs: List[Document]
    formatted_context: str
    
class RouteDecisionDTO(BaseModel):
    """DTO for routing decisions."""
    route: Literal["EXPERT_RAG", "USER_MEMORY", "CHITCHAT"] = Field(
        description="La categoría exacta a la que pertenece la consulta del usuario."
    )