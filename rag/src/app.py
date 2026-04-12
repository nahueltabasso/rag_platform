from typing import Dict
from rag_system import process_query, get_retriever_info
import streamlit as st
import argparse
import json

def main(config: Dict):
    # Page Set-Up
    st.set_page_config(
        # page_title="Sistema RAG - Asistente Segunda Guerra Mundial",
        page_title=config.get("name", "Sistema RAG"),
        page_icon=config.get("page_icon"),
        layout="wide"
    )

    # Títle
    st.title(config.get("name", "Sistema RAG - Asistente Segunda Guerra Mundial"))
    st.divider()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("📋 Información del Sistema")
        
        # Información del retriever
        retriever_info = get_retriever_info()
        
        st.markdown("**🔍 Retriever:**")
        st.info(f"Tipo: {retriever_info['tipo']}")
        
        st.markdown("**🤖 Modelos:**")
        st.info("Consultas: GPT-4o-mini\nRespuestas: GPT-4o")
        
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
    if prompt := st.chat_input("Escribe tu consulta sobre la segunda guerra mundial..."):
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generar respuesta
        with st.spinner("🔍 Analizando..."):
            print(f"Processing query: {prompt}")
            response, docs = process_query(config, prompt)
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

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # print(f"Loaded config: {config}")
    print(config.get('vector_db', {}).get('embbedings_model'))
    
    main(config=config)