from rag.src.agent_engine import MemoryRAGAgentEngine
from rag_system import get_rag_service

CONFIG_PATH = "rag/config/wwII_wiki_config.json"

rag_service = get_rag_service(CONFIG_PATH)

engine = MemoryRAGAgentEngine(user_id="test_user", rag_service=rag_service)
workflow = engine.workflow

session_id = "test_session"

while True:
    try:
        user_input = input("Human: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break
    
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting...")
        break
    
    response = engine.chat(message=user_input, thread_id=session_id)
    
    print(f"Assistant: {response['response']}\n\n") # type: ignore
    
    
    
# USER INPUT --> QUERY
# CARGA DEL HISTORIAL DE LA SESION
# REFORMULACION DE LA PREGUNTA
# GENERACION DE PREGUNTAS EN BASE A LA PREGUNTA REFORMULADA
# OBTENCION DE DOCUMENTOS RELEVANTES
# FILTRADO DE DOCUMENTOS RELEVANTES EN BASE A LA PREGUNTA REFORMULADA
# CONSTRUIR CONTEXTO
# GENERACION DE RESPUESTA EN BASE A LOS DOCUMENTOS RELEVANTES
# ACTUALIZACION DEL HISTORIAL DE LA SESION
# RESPUESTA AL USUARIO
    
        