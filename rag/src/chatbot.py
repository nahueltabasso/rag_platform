from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from memory_manager import ModernMemoryManager, MemoryState
import sqlite3
import os

class ModernChatBot:
    
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.memory_manager = ModernMemoryManager(user_id=user_id)
        
        # LLM model configuration
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        
        # System Template with dynamic context
        self.system_template = """
            Eres un asistente personal inteligente y amigable.
            
            Caracteristicas de tu personalidad:
            - Eres util, empatico y conversacional
            - Recuerdas informacion importante de conversaciones anteriores
            - Adaptas tu estilo a las preferencias del usuario
            - Eres proactivo ofreciendo sugerencias relevantes
            - Mantienes un tono profesonal pero cercano
            
            {context}
            
            Usa esta informacion para personalizar tus respuestas, pero no menciones explicitamente que tienes memoria
            a menos que sea relevante para la conversacion.
        """
        
        # Message trimming configuration to manage token limits
        self.message_trimeer = trim_messages(
            strategy="last",
            max_tokens=4000,
            token_counter=self.llm,
            start_on="human",
            include_system=True
        )
        
        # Create a LangGraph App
        self.app = self._create_app()
        
    def _create_app(self):
        """Create a LangGraph app with extended state."""
        workflow = StateGraph(state_schema=MemoryState)
        
        # Configure the workflow with nodes
        workflow.add_node("memory_retrieval", self.memory_retrieval_node)
        workflow.add_node("context_optimization", self.context_optimization_node)
        workflow.add_node("response_generation", self.response_generation_node)
        workflow.add_node("memory_extraction", self.memory_extraction_node)
        
        # Define the flow of nodes
        workflow.add_edge(START, "memory_retrieval")
        workflow.add_edge("memory_retrieval", "context_optimization")
        workflow.add_edge("context_optimization", "response_generation")
        workflow.add_edge("response_generation", "memory_extraction")
        workflow.add_edge("memory_extraction", END)
        
        # Persistence with SQLite
        db_path = os.path.join(self.memory_manager.user_dir, "langgraph_memory.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def chat(self, user_message: str, chat_id: str = "default") -> Dict:
        """Send message and retrieve a response from chatbot."""
        try:
            # Thread config for chat
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}
            
            # Update title with user message if is necessary
            chat_info = self.memory_manager.get_chat_info(chat_id)
            if chat_info['title'] == "Nuevo chat": # type: ignore
                chat_title = self.memory_manager._generate_chat_title(first_message=user_message)
                self.memory_manager.update_chat_metadata(chat_id=chat_id, title=chat_title)
                
            # Invoke chatbot with new meessage
            result = self.app.invoke({"messages": [HumanMessage(content=user_message)]}, config=config) # type: ignore
            
            # Extract the chatbot response
            assistant_response = result['messages'][-1].content
            return {
                "success": True,
                "response": assistant_response,
                "error": None,
                "memories_used": len(result.get('vector_memories', [])),
                "context_optimized": True
            }
        except Exception as e:
            print(f"Error in chatbot invocation: {e}")
            return {
                "success": False,
                "response": None,
                "error": str(e),
                "memories_used": 0,
                "context_optimized": False
            }
            
    def get_chat_history(self, chat_id: str="default", limit: int=50) -> List:
        """Get chat history for a given chat ID. using the LangGraph state persistence."""
        try:
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}
            
            # Obtain the state history from the checkpointer
            state = self.app.get_state(config=config) # type: ignore
            
            if not state.values or "messages" not in state.values:
                return []
            
            messages = state.values['messages']
            # Convert to a UI format
            history = []
            for msg in messages[-limit:]:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    history.append({
                        'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', None) or "unknown"
                    })
            return history
        except Exception as e:
            print(f"Error retrieving chat history: {e}")
            return []
        
    def clear_chat(self, chat_id: str="default") -> bool:
        """Clear chat history."""
        try:
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}
            self.app.invoke({"messages": []}, config=config) # type: ignore
            return True
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
        
    def delete_chat_from_langgraph(self, chat_id: str="default") -> bool:
        """Delete chat from LangGraph persistence."""
        try:
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}
            return True
        except Exception as e:
            print(f"Error deleting chat from LangGraph: {e}")
            return False
    
    # Retrieve memory node
    def memory_retrieval_node(self, state) -> Dict:
        """Node to retrieve relevant memories."""
        messages = state['messages']
        
        if not messages:
            return {"vector_memories": []}
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        if not last_user_message:
            return {"vector_memories": []}
        
        # Retrieve relevant memories based on the last user message
        relevant_memories = self.memory_manager.search_vector_memory(
            query=last_user_message.content # type: ignore
        )
        
        return {"vector_memories": relevant_memories}
    
    # Optimize context node
    def context_optimization_node(self, state) -> Dict:
        """Node to optimize context using trim_messages."""
        messages = state['messages']
        
        # Apply smart trimming to manage token limits
        trimmed_messages = self.message_trimeer.invoke(messages)
        
        return {"messages": trimmed_messages}
        
    # Response generation node
    def response_generation_node(self, state) -> Dict:
        """Node to generate response using optimized context."""
        messages = state['messages']
        vector_memories = state.get('vector_memories', [])
        
        if not messages:
            return {"messages": []}
        # Build system context with retrieved memories
        if vector_memories:
            context_parts = ["Informacion relevante que recuerdas del usuario:"]
            for memory in vector_memories:
                context_parts.append(f" - {memory}")
                
            context = "\n".join(context_parts)
        else:
            context = "No tienes recuerdos relevantes del usuario."
            
        # Create a prompt with dynamic system context
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template.format(context=context)),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Generate response using the LLM
        chain = prompt | self.llm
        response = chain.invoke({"messages": messages})
        
        return {"messages": response}
    
    # Memory update node
    def memory_extraction_node(self, state) -> Dict:
        """Node to extract and update new relevant memories."""
        messages = state['messages']
        last_extraction = state.get('last_memory_extraction', "")
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        if not last_user_message:
            return {}
        
        # Extract only if the user message is different from the last extraction to avoid duplicates
        if last_extraction != last_user_message.content:
            self.memory_manager.extract_and_store_memories(user_message=last_user_message.content) # type: ignore
            return {"last_memory_extraction": last_user_message.content}
        else:
            return {}
        
class ChatBotManager:
    """Manager to handle multiple chatbots for different users."""


    _instances = {}        

    @classmethod
    def get_chatbots(cls, user_id):
        """Get or create a chatbot instance for a user."""
        if user_id not in cls._instances:
            cls._instances[user_id] = ModernChatBot(user_id)
        return cls._instances[user_id]
    
    @classmethod
    def remove_chatbot(cls, user_id):
        """Remove a chatbot instance for a user."""
        if user_id in cls._instances:
            del cls._instances[user_id]
            
    @classmethod
    def clear_all(cls):
        """Clear all chatbot instances."""
        cls._instances.clear()