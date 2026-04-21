# app.py
import streamlit as st
from retriever import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage 

# ==========================================
# 1. PAGE SETUP & CSS INJECTION
# ==========================================
st.set_page_config(page_title="MedlinePlus AI", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
        background-color: #E0ECFF;
        border-radius: 15px;
        padding: 1rem;
        margin-left: auto;
        max-width: 80%;
        border: 1px solid #E2E8F0;
    }
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 1rem;
        margin-right: auto;
        max-width: 90%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    .stChatMessageAvatar { display: none; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD BACKEND
# ==========================================
@st.cache_resource(show_spinner="Loading Medical Database...")
def load_backend():
    # Streamlit runs this once and caches the engine
    return get_rag_chain()

rag_chain = load_backend()

# ==========================================
# 3. UI: SIDEBAR & CHAT STATE
# ==========================================
with st.sidebar:
    st.title("🩺 MedlinePlus AI")
    st.caption("A private, locally-hosted medical RAG assistant.")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("⚠️ **Disclaimer:** This AI provides informational content only and is not a substitute for professional medical advice. Always consult a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🧾 View Sources"):
                for url in msg["sources"]:
                    st.markdown(f"- [{url}]({url})")

# ==========================================
# 4. UI: INPUT & INTERACTION
# ==========================================
if user_query := st.chat_input("Ask a medical question..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical database..."):
            try:
                # Call the backend engine
                langchain_history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_history.append(HumanMessage(content=msg["content"]))
                    else:
                        langchain_history.append(AIMessage(content=msg["content"]))
                response = rag_chain.invoke({"input": user_query, "chat_history": langchain_history})
                st.write("🔍 DEBUG - Context retrieved:", response["context"])
                answer = response["answer"]
                
                sources = set()
                for doc in response["context"]:
                    if "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                
                st.markdown(answer)
                if sources:
                    with st.expander("🧾 View Sources"):
                        for url in sources:
                            st.markdown(f"- [{url}]({url})")
                            
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": list(sources)
                })
                
            except Exception as e:
                st.error("Something went wrong. Please try again.")
                print(f"Error: {e}")