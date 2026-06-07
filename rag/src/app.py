from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import json

import streamlit as st

from rag.src.chatbot_manager import ChatBotManager
from rag.src.rag_system import RAGService, get_rag_service
from rag.src.user_manager import UserManager


_SHARED_RAG_SERVICE: Optional[RAGService] = None


def get_config_files() -> List[str]:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    return sorted(str(path.resolve()) for path in config_dir.iterdir() if path.is_file())


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_shared_rag_service(config_path: str) -> RAGService:
    global _SHARED_RAG_SERVICE

    normalized_path = str(Path(config_path).resolve())
    if _SHARED_RAG_SERVICE is None:
        _SHARED_RAG_SERVICE = get_rag_service(normalized_path)
    elif _SHARED_RAG_SERVICE.config_path != normalized_path:
        _SHARED_RAG_SERVICE.set_config(normalized_path)

    return _SHARED_RAG_SERVICE


def build_initial_messages(config: dict) -> List[Dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": config.get("initial_message", "Hola, ¿en qué puedo ayudarte?"),
            "meta": {"timestamp": get_current_time_label()},
        }
    ]


def get_current_time_label() -> str:
    return datetime.now().strftime("%H:%M")


def init_session_state(default_config_path: str) -> None:
    if "selected_config_path" not in st.session_state:
        st.session_state.selected_config_path = str(Path(default_config_path).resolve())
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "active_chat_by_user" not in st.session_state:
        st.session_state.active_chat_by_user = {}
    if "ui_chat_messages" not in st.session_state:
        st.session_state.ui_chat_messages = {}


def is_valid_user_id(user_id: str) -> bool:
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return bool(user_id) and all(char in allowed_chars for char in user_id)


def get_chatbot(user_id: str, rag_service: RAGService):
    return ChatBotManager.get_chatbot(user_id=user_id, rag_service=rag_service)


def create_new_chat(user_id: str, chatbot, config: dict) -> str:
    chat_id = chatbot.memory_manager.create_new_chat()
    user_chats = st.session_state.ui_chat_messages.setdefault(user_id, {})
    user_chats[chat_id] = build_initial_messages(config)
    st.session_state.active_chat_by_user[user_id] = chat_id
    return chat_id


def ensure_user_chat_state(user_id: str, chatbot, config: dict) -> str:
    user_chats = st.session_state.ui_chat_messages.setdefault(user_id, {})
    chats = chatbot.memory_manager.get_user_chats()

    if not chats:
        return create_new_chat(user_id=user_id, chatbot=chatbot, config=config)

    active_chat_id = st.session_state.active_chat_by_user.get(user_id)
    known_chat_ids = [chat["chat_id"] for chat in chats]

    if active_chat_id not in known_chat_ids:
        active_chat_id = chats[0]["chat_id"]
        st.session_state.active_chat_by_user[user_id] = active_chat_id

    if active_chat_id not in user_chats:
        user_chats[active_chat_id] = build_initial_messages(config)

    return active_chat_id


def reset_ui_state_for_config_change() -> None:
    st.session_state.active_chat_by_user = {}
    st.session_state.ui_chat_messages = {}
    ChatBotManager.clear_all_chatbots()


def render_config_sidebar(config_files: List[str], rag_service: RAGService) -> dict:
    selected_file = st.selectbox(
        "Archivo de configuración",
        options=config_files,
        index=next(
            (
                index for index, path in enumerate(config_files)
                if path == st.session_state.selected_config_path
            ),
            0,
        ),
        format_func=lambda path: Path(path).name,
    )

    selected_file_path = str(Path(selected_file).resolve())
    if selected_file_path != st.session_state.selected_config_path:
        rag_service.set_config(selected_file_path)
        st.session_state.selected_config_path = selected_file_path
        reset_ui_state_for_config_change()
        st.rerun()

    return load_config(st.session_state.selected_config_path)


def render_user_sidebar(rag_service: RAGService):
    st.markdown("**Usuarios**")
    existing_users = UserManager.get_users()

    if existing_users:
        selected_user = st.selectbox(
            "Usuario activo",
            options=existing_users,
            index=existing_users.index(st.session_state.current_user)
            if st.session_state.current_user in existing_users
            else 0,
        )
        if selected_user != st.session_state.current_user:
            st.session_state.current_user = selected_user
            st.rerun()
    else:
        st.info("No hay usuarios creados todavía.")

    with st.expander("Crear usuario", expanded=not existing_users):
        new_user_id = st.text_input("ID de usuario", placeholder="usuario_1")
        if st.button("Crear usuario", type="primary", use_container_width=True):
            if not is_valid_user_id(new_user_id):
                st.error("Usa solo letras, números, guion y guion bajo.")
            elif UserManager.exists_user(new_user_id):
                st.error("Ese usuario ya existe.")
            elif UserManager.create_user(new_user_id):
                st.session_state.current_user = new_user_id
                ChatBotManager.remove_chatbot(new_user_id)
                get_chatbot(new_user_id, rag_service)
                st.rerun()
            else:
                st.error("No se pudo crear el usuario.")


def render_chat_sidebar(user_id: str, chatbot, config: dict) -> str:
    active_chat_id = ensure_user_chat_state(user_id=user_id, chatbot=chatbot, config=config)
    chats = chatbot.memory_manager.get_user_chats()

    st.markdown("**Chats**")
    selected_chat_id = st.selectbox(
        "Chat activo",
        options=[chat["chat_id"] for chat in chats],
        index=next(
            (index for index, chat in enumerate(chats) if chat["chat_id"] == active_chat_id),
            0,
        ),
        format_func=lambda chat_id: next(
            (chat["title"] for chat in chats if chat["chat_id"] == chat_id),
            chat_id,
        ),
    )

    if selected_chat_id != active_chat_id:
        st.session_state.active_chat_by_user[user_id] = selected_chat_id
        ensure_user_chat_state(user_id=user_id, chatbot=chatbot, config=config)
        st.rerun()

    if st.button("Nuevo chat", type="primary", use_container_width=True):
        create_new_chat(user_id=user_id, chatbot=chatbot, config=config)
        st.rerun()

    if st.button("Reiniciar conversación", use_container_width=True):
        create_new_chat(user_id=user_id, chatbot=chatbot, config=config)
        st.rerun()

    return st.session_state.active_chat_by_user[user_id]


def render_system_info(config: dict) -> None:
    st.markdown("**Sistema**")
    retriever_info = "MMR + Multiquery Hybrid" if config.get("hybrid_search", {}).get("enable") else "MMR"
    st.info(f"Tema: {config.get('topic', 'N/A')}")
    st.info(f"Retriever: {retriever_info}")
    st.info(
        "Consultas: "
        f"{config.get('models', {}).get('query_model', 'N/A')}\n"
        "Respuestas: "
        f"{config.get('models', {}).get('generation_model', 'N/A')}"
    )


def render_chat_messages(messages: List[Dict[str, object]]) -> None:
    for message in messages:
        role = str(message["role"])
        with st.chat_message(role):
            st.markdown(str(message["content"]))
            meta = message.get("meta", {})
            if isinstance(meta, dict):
                caption_parts = []
                timestamp = meta.get("timestamp")
                if timestamp:
                    caption_parts.append(str(timestamp))
                if role == "assistant":
                    memories_used = meta.get("memories_used")
                    if memories_used:
                        caption_parts.append(f"Memorias usadas: {memories_used}")
                    if meta.get("context_optimized"):
                        caption_parts.append("Contexto optimizado")
                if caption_parts:
                    st.caption(" | ".join(caption_parts))


def render_relevant_documents(messages: List[Dict[str, object]]) -> None:
    st.markdown("### Documentos Relevantes")

    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not assistant_messages:
        st.info("Todavía no hay respuestas con documentos para mostrar.")
        return

    last_assistant_message = assistant_messages[-1]
    meta = last_assistant_message.get("meta", {})
    docs = meta.get("docs", []) if isinstance(meta, dict) else []

    if not docs:
        st.info("La última respuesta no recuperó documentos relevantes.")
        return

    for doc in docs:
        chunk = doc.get("chunk", "?")
        with st.expander(f"Fragmento {chunk}", expanded=False):
            st.markdown(f"**Fuente:** {doc.get('url', 'Sin fuente')}")
            st.markdown("**Contenido:**")
            st.text(doc.get("content", ""))


def main(config_path: str):
    init_session_state(default_config_path=config_path)
    rag_service = get_shared_rag_service(st.session_state.selected_config_path)
    config_files = get_config_files()
    config = load_config(st.session_state.selected_config_path)

    st.set_page_config(
        page_title=config.get("name", "Sistema RAG"),
        page_icon=config.get("page_icon"),
        layout="wide",
    )

    st.title(config.get("name", "Sistema RAG"))
    st.divider()

    with st.sidebar:
        config = render_config_sidebar(config_files=config_files, rag_service=rag_service)
        st.divider()
        render_user_sidebar(rag_service=rag_service)
        st.divider()

        current_user = st.session_state.current_user
        if current_user:
            chatbot = get_chatbot(current_user, rag_service)
            active_chat_id = render_chat_sidebar(user_id=current_user, chatbot=chatbot, config=config)
            st.divider()
            render_system_info(config)
        else:
            active_chat_id = None

    current_user = st.session_state.current_user
    if not current_user:
        st.info("Selecciona o crea un usuario para comenzar a usar el orquestador.")
        return

    chatbot = get_chatbot(current_user, rag_service)
    active_chat_id = active_chat_id or ensure_user_chat_state(user_id=current_user, chatbot=chatbot, config=config)
    active_messages = st.session_state.ui_chat_messages[current_user][active_chat_id]
    current_chat_info = chatbot.memory_manager.get_chat_info(active_chat_id)

    st.subheader(current_chat_info["title"] if current_chat_info else "Chat")

    chat_col, docs_col = st.columns([2, 1])
    with chat_col:
        render_chat_messages(active_messages)
    with docs_col:
        render_relevant_documents(active_messages)

    input_legend = config.get(
        "input_field_legend",
        config.get("input_field_leyend", "Escribe tu pregunta aquí..."),
    )
    if prompt := st.chat_input(input_legend):
        active_messages.append(
            {
                "role": "user",
                "content": prompt,
                "meta": {"timestamp": get_current_time_label()},
            }
        )
        with st.spinner("Pensando..."):
            result = chatbot.chat(message=prompt, thread_id=active_chat_id)

        if result["success"]:
            active_messages.append(
                {
                    "role": "assistant",
                    "content": result["response"],
                    "meta": {
                        "docs": result["docs"],
                        "timestamp": get_current_time_label(),
                        "memories_used": result["memories_used"],
                        "context_optimized": result["context_optimized"],
                    },
                }
            )
            chatbot.memory_manager.update_chat_metadata(active_chat_id, increment_messages=True)
        else:
            st.error(result["error"])

        st.rerun()

    st.divider()
    st.markdown(
        f"<div style='text-align: center; color: #666;'>{config.get('footer_message', 'Asistente')}</div>",
        unsafe_allow_html=True,
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    args = get_args()
    config_path = args.config_path
    
    if not config_path:
        config_path = "./config/wwII_wiki_config.json"

    main(config_path=config_path)