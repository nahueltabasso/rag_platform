from typing import List
from pathlib import Path
from rag_system import get_rag_service
import streamlit as st
import argparse
import json
import os


def get_config_files() -> List[str]:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    return [str(path) for path in config_dir.iterdir() if path.is_file()]

def main(config_path: str):
    current_config_path = str(Path(config_path).resolve())
    config_files = get_config_files()

    if "selected_config_path" not in st.session_state:
        st.session_state.selected_config_path = current_config_path

    current_config_path = st.session_state.selected_config_path

    with open(current_config_path, 'r') as f:
        config = json.load(f)

    rag_service = get_rag_service(config_path=current_config_path)
    
    # Page Set-Up
    st.set_page_config(
        page_title=config.get("name", "Sistema RAG"),
        page_icon=config.get("page_icon"),
        layout="wide"
    )

    # Títle
    st.title(config.get("name", "Sistema RAG"))
    st.divider()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": config.get("initial_message", "Hola, ¿en qué puedo ayudarte?")
            }
        ]
    # Sidebar
    with st.sidebar:
        selected_file = st.selectbox(
            "📁 Archivo de configuración",
            options=config_files,
            index=next(
                (
                    i for i, path in enumerate(config_files)
                    if str(Path(path).resolve()) == st.session_state.selected_config_path
                ),
                0
            ),
            format_func=lambda path: Path(path).name,
        )

        selected_file_path = str(Path(selected_file).resolve())

        if selected_file_path != st.session_state.selected_config_path:
            rag_service.set_config(selected_file_path)
            st.session_state.selected_config_path = selected_file_path

            with open(selected_file_path, 'r') as f:
                new_config = json.load(f)

            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": new_config.get("initial_message", "Hola, ¿en qué puedo ayudarte?")
                }
            ]
            st.rerun()
        
        # Información del retriever
        st.markdown("**🔍 Retriever:**")
        st.info(f"Tema: {config.get('topic', 'N/A')}")
        retriever_info = "MMR + Multiquery Hybrid" if config.get('hybrid_search', {}).get('enable') else "MMR"
        st.info(f"Tipo: {retriever_info}")
        
        st.markdown("**🤖 Modelos:**")
        st.info(f"Consultas: {config.get('models', {}).get('query_model', 'N/A')}\nRespuestas: {config.get('models', {}).get('generation_model', 'N/A')}")

        st.markdown("**📄 Base de Datos Vectorial:**")
        st.info(f"Vector DB: {config.get('vector_db', {}).get('type')}")

        st.divider()
        
        if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main layout with columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Chat")
        
        # Mostrar historial de mensajes
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with col2:
        st.markdown("### 📄 Documentos Relevantes")
        
        # Mostrar documentos de la última consulta
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant" and "docs" in last_message:
                docs = last_message["docs"]
                
                if docs:
                    for doc in docs:
                        with st.expander(f"📄 Fragmento {doc['chunk']}", expanded=False):
                            st.markdown(f"**Fuente:** {doc['url']}")
                            st.markdown("**Contenido:**")
                            st.text(doc['content'])

    # User Input
    if prompt := st.chat_input(config.get("input_field_leyend", "Escribe tu pregunta aquí...")):
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generar respuesta
        with st.spinner("🔍 Analizando..."):
            print(f"Processing query: {prompt}")
            response, docs = rag_service.process_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})
        
        # Recargar para mostrar los nuevos mensajes
        st.rerun()

    # Footer
    st.divider()
    st.markdown(
        f"<div style='text-align: center; color: #666;'>{config.get('footer_message', 'Asistente Legal')}</div>", 
        unsafe_allow_html=True
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