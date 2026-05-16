from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

# Extended State which combine messages with vector memory
class AppState(TypedDict):
    """State that combines LangGraph messages with vector memory."""
    messages: Annotated[List[BaseMessage], add_messages]
    vector_memories: List[str] # Vector memories actives IDs
    user_profile: Dict[str, Any] # User profile information
    last_memory_extraction: Optional[str] # Last processed message for memory extraction

    rewritten_query: str
    context_docs: List[Document]
    