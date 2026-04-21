import json
from rag.src.config_schema import RAGConfig
from rag_system import get_rag_service

CONFIG_PATH = "rag/config/wwII_wiki_config.json"

rag_service = get_rag_service(CONFIG_PATH)

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
    
    response, _ = rag_service.process_query(query=user_input, session_id=session_id)
    print(f"Assistant: {response}\n\n") # type: ignore
    
        