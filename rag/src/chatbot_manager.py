from rag.src.agent_engine import MemoryRAGAgentEngine
from rag.src.rag_system import RAGService

class ChatBotManager:
    
    _instances = {}
    
    @classmethod
    def get_chatbot(cls, user_id: str, rag_service: RAGService):
        if user_id not in cls._instances:
            cls._instances[user_id] = MemoryRAGAgentEngine(user_id=user_id, rag_service=rag_service)
        return cls._instances[user_id]
    
    @classmethod
    def remove_chatbot(cls, user_id: str):
        if user_id in cls._instances:
            del cls._instances[user_id]
            
    @classmethod
    def clear_all_chatbots(cls):
        cls._instances.clear()