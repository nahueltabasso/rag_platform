from typing import List, Dict, Any, Optional
from datetime import datetime
from typing_extensions import TypedDict, Annotated
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from config import CATEGORY_DESCRIPTION, CONTENT_DESCRIPTION, IMPORTANCE_DESCRIPTION, USERS_DIR, MAX_VECTOR_RESULTS 
import chromadb
import os
import uuid
import json

# Extended State which combine messages with vector memory
class MemoryState(TypedDict):
    """State that combines LangGraph messages with vector memory."""
    messages: Annotated[List[BaseMessage], add_messages]
    vector_memories: List[str] # Vector memories actives IDs
    user_profile: Dict[str, Any] # User profile information
    last_memory_extraction: Optional[str] # Last processed message for memory extraction
    
class ExtractedMemory(BaseModel):
    """Model for structured memory extracted from conversations."""
    category: str = Field(description=CATEGORY_DESCRIPTION)
    content: str = Field(description=CONTENT_DESCRIPTION)
    importance: int = Field(description=IMPORTANCE_DESCRIPTION, ge=1, le=5)
    
    
class ModernMemoryManager:
    
    def __init__(self, user_id: str="deafault_id") -> None:
        self.user_id = user_id
        self.user_dir = os.path.join(USERS_DIR, self.user_id)
        os.makedirs(self.user_dir, exist_ok=True)
        
        # Init ChromaDB vector store for transversal memory
        self.chromadb_user_path = os.path.join(self.user_dir, "chroma_db")
        self._init_vector_db()
        
        # Smart memory extraction system
        self._init_extraction_system()
        
        # Path to LangGraph database
        self.langgraph_db_path = os.path.join(self.user_dir, "langgraph_memory.db")
        
    def _init_vector_db(self) -> None:
        """Initialize ChromaDB Vector Store."""
        try:
            self.vector_store = Chroma(
                collection_name=f"memory_{self.user_id}",
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
                persist_directory=self.chromadb_user_path
            )
            self.client = chromadb.PersistentClient(path=self.chromadb_user_path)
            try:
                self.collection = self.client.get_collection(name=f"memory_{self.user_id}")
            except Exception as e:
                self.collection = self.client.create_collection(name=f"memory_{self.user_id}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store = None
            self.collection = None
            
    def _init_extraction_system(self) -> None:
        """Initialize the smart memory extraction system."""
        try:
            self.extraction_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
            self.memory_parser = PydanticOutputParser(pydantic_object=ExtractedMemory)
            self.extraction_template = PromptTemplate(
                template="""Analiza el siguiente mensaje del usuario y determina si contiene informacion importante que
                deba recordarse para futuras conversaciones.
                
                Categorias Disponibles:
                - personal: Nombre, edad, ubicacion, familia, etc.add()
                - profesional: Trabajo, empresa, proyectos, habilidades, estudios
                - preferencias: Gustos, disgustos, preferencias personales
                - hechos_importantes: Informacion relevante que debe recordarse
                
                Mensaje del usuario: "{user_message}"
                
                Si el mensaje contiene informacion importante, extrae UNA memoria (la mas imortante).
                Si no contiene informacion relevante para recordar responde con categoria "none".
                
                {format_instructions}""",
                input_variables=["user_message"],
                partial_variables={"format_instructions": self.memory_parser.get_format_instructions()}
            )
            
            self.extraction_chain = self.extraction_template | self.extraction_llm | self.memory_parser
        except Exception as e:
            print(f"Error initializing extraction system: {e}")
            self.extraction_chain = None
    
    # === CHATS MANAGEMENT (Hybrid: JSON + LangGraph) ===
    
    def get_user_chats(self) -> List:
        """Retrieve all chat sessions for the user."""
        try:
            # If not exists metadata file, return empty
            chats_meta_file = os.path.join(self.user_dir, "chat_meta.json")
            if not os.path.exists(chats_meta_file):
                return []
            
            with open(chats_meta_file, 'r', encoding='utf-8') as f:
                chats_data = json.load(f)
            
            # Sort chats by last updated
            chats_data.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            return chats_data
        except Exception as e:
            print(f"Error retrieving user chats: {e}")
            return []
        
    def _save_chats_metadata(self, chats_data: List) -> None:
        """Save the list of chat sessions to the metadata file."""
        try:
            chats_meta_file = os.path.join(self.user_dir, "chat_meta.json")
            with open(chats_meta_file, 'w', encoding='utf-8') as f:
                json.dump(chats_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving chats metadata: {e}")
        
    def create_new_chat(self, first_message: str=""):
        """Create a new chat session and return its ID."""
        chat_id = str(uuid.uuid4())
        
        # Generate title base on first message
        title = self._generate_chat_title(first_message) if first_message else "Nuevo Chat"
        # Create metadata entry
        new_chat = {
            'chat_id': chat_id,
            'title': title,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            "message_count": 0
        }
        
        # Load existing chats, append new one and save
        chats_data = self.get_user_chats()
        chats_data.append(new_chat)
        self._save_chats_metadata(chats_data)
        return chat_id
    
    def update_chat_metadata(self, chat_id: str, title: str=None, increment_messages: bool=False) -> None: # type: ignore
        """Update chat metadata."""
        chats_data = self.get_user_chats()
        for chat in chats_data:
            if chat['chat_id'] == chat_id:
                if title:
                    chat['title'] = title
                if increment_messages:
                    chat['message_count'] = chat.get('message_count', 0) + 1
                chat['updated_at'] = datetime.now().isoformat()
                break
        else:
            # If not exists, create new chat metadata
            if chat_id:
                new_chat = {
                    'chat_id': chat_id,
                    'title': title or 'Chat sin titulo',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'message_count': 1 if increment_messages else 0
                }
                chats_data.append(new_chat)
        self._save_chats_metadata(chats_data)
        
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session."""
        try:
            chats_data = self.get_user_chats()
            chats_data = [chat for chat in chats_data if chat['chat_id'] != chat_id]
            self._save_chats_metadata(chats_data)
            return True
        except Exception as e:
            print(f"Error deleting chat: {e}")
            return False
        
    def get_chat_info(self, chat_id) -> Optional[Dict[str, Any]]:
        chats = self.get_user_chats()
        for chat in chats:
            if chat['chat_id'] == chat_id:
                return chat
        return None
    
    def _generate_chat_title(self, first_message: str) -> str:
        """Generate a chat title based on the first message."""
        try:
            if not self.extraction_llm:
                return first_message[:30] + "..." if len(first_message) > 30 else first_message
            
            title_prompt = PromptTemplate(
                template="""Genera un titulo breve (maximo 4-5 palabras) para una conversacion que empieza con este mensaje:
                    {message}
                    El titulo debe:
                    - Ser conciso y descriptivo
                    - Capturar el tema principal
                    - Ser apropiado para un historial de chat
                    - No incluir comillas
                    
                    Titulo:""",
                input_variables=["message"]
            )
            title_chain = title_prompt | self.extraction_llm
            response = title_chain.invoke({"message": first_message})
            title = response.strip().strip('"').strip("'") # type: ignore
            return title if len(title) < 50 else title[:47] + "..."
        except Exception as e:
            print(f"Error generating chat title: {e}")
            return first_message[:30] + "..." if len(first_message) > 30 else first_message
        
    # === VECTOR MEMORY MANAGEMENT ===
    def save_vector_memory(self, content: str, metadata: Optional[Dict]=None) -> str:
        """Save a memory to the vector store."""
        if not self.collection:
            print("Vector store not initialized.")
            return ""
        
        try:
            memory_id = str(uuid.uuid4())
            doc_metadata = metadata or {}
            doc_metadata.update({
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "memory_id": memory_id
            })
            
            self.collection.add(
                documents=[content],
                ids=[memory_id],
                metadatas=[doc_metadata]
            )
            return memory_id
        except Exception as e:
            print(f"Error saving vector memory: {e}")
            return ""
        
    def search_vector_memory(self, query: str, k: int=MAX_VECTOR_RESULTS) -> List[Dict]:
        """Search relevant memories in the vector store."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            return results.get('documents', [[]])[0] if results else [] # type: ignore
        except Exception as e:
            print(f"Error searching vector memory: {e}")
            return []
        
    def get_all_vector_memories(self) -> List[Dict]:
        """Retrieve all memories from the vector store."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get()
            memories = []
            
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    memory = {
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i] if results['metadatas'] else {} # type: ignore
                    }
                    memories.append(memory)
            return memories
        except Exception as e:
            print(f"Error retrieving all vector memories: {e}")
            return []
        
    # === SMART MEMORY EXTRACTION ===
    def extract_and_store_memories(self, user_message: str) -> bool:
        """Extract important information from a user message and store it as memory if relevant."""
        if not self.extraction_chain:
            return self._extract_memories_manual(user_message=user_message)
        
        try:
            extracted_memory = self.extraction_chain.invoke({"user_message": user_message})
            
            if extracted_memory.category.lower() != "none" and extracted_memory.importance >= 2:
                memory_id = self.save_vector_memory(
                    content=extracted_memory.content,
                    metadata={
                        "category": extracted_memory.category,
                        "importance": extracted_memory.importance,
                        'original_message': user_message
                    }
                )
                return bool(memory_id)
            return False
        except Exception as e:
            print(f"Error extracting and storing memory: {e}")
            return self._extract_memories_manual(user_message=user_message)
        
        
    def _extract_memories_manual(self, user_message: str) -> bool:
        """Manual method to extract memories without LLM, for fallback."""
        message_lower = user_message.lower()
        rules = [
            (["me llamo", "mi nombre es", "soy"], "personal", f"Info Personal: {user_message}"),
            (["trabajo en", "trabajo como", "mi profesion"], "profesional", f"Info Profesional: {user_message}"),
            (["me gusta", "me encanta", "prefiero", "odio"], "preferencias", f"Info Preferencias: {user_message}"),
            (["importante", "recuerda que", "no olvides"], "hechos_importantes", f"Hecho Importante: {user_message}")
        ]
        
        for phrases, category, memory_text in rules:
            if any (phrase in message_lower for phrase in phrases):
                memory_id = self.save_vector_memory(content=memory_text, metadata={"category": category})
                print(f"Memoria extraida manualmente: {memory_text} (Categoria: {category})")
                return bool(memory_id)
        return False
    