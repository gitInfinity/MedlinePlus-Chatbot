import streamlit as st
from retriever import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage 

# ==========================================
# 1. SETUP & THEME (The CSS Alignment Fix)
# ==========================================
st.set_page_config(page_title="MedlinePlus AI", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF !important; }
    
    /* 1. Global Message Container */
    [data-testid="stChatMessage"] { 
        padding: 1.2rem !important;
        border-radius: 20px !important;
        margin-bottom: 1.5rem !important;
        max-width: 85% !important;
        display: flex !important;
    }
    
    header, [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        backdrop-filter: none !important; /* Removes the blurry effect */
    }

    /* 2. Ensure the icons (Settings, Deploy) stay visible */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        color: #1976D2 !important; /* Optional: Makes icons match your blue theme */
    }

    /* 3. Pull content up slightly so it feels more integrated */
    .block-container {
        padding-top: 3rem !important;
    }

    /* 🔵 ASSISTANT (Align Left) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #E3F2FD !important;
        border-left: 5px solid #1976D2 !important;
        margin-right: auto !important;
        flex-direction: row !important;
    }

    /* ⚪ USER (Align Right) */
    /* We target the container that holds the user icon/label */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-right: 5px solid #cbd5e1 !important;
        margin-left: auto !important; /* The magic bullet for right-alignment */
        flex-direction: row-reverse !important; /* Flips the bubble to the right */
        border-left: none !important;
    }

    /* Hide Avatars for the minimal look */
    [data-testid="stChatMessageAvatar"] { display: none; }
    
    /* Ensure text aligns correctly inside reversed bubbles */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
        text-align: left !important;
    }

    /* Buttons & Sidebar UI */
    .stButton > button {
        border-radius: 12px !important;
        background-color: white !important;
        color: #1976D2 !important;
        border: 1px solid #BBDEFB !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_backend():
    return get_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. THE RESPONSE FUNCTION (Fixed Persistence)
# ==========================================
def handle_query(query_text):
    # 1. Save to session state so it doesn't disappear on rerun
    st.session_state.messages.append({"role": "user", "content": query_text})
    
    # 2. Manually render the message right now so it stays while AI thinks
    # We do this OUTSIDE the loop to prevent "double rendering" later
    with st.chat_message("user"):
        st.markdown(query_text)
    
    # 3. Process AI Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting MedlinePlus database..."):
            chain = load_backend()
            history = [
                HumanMessage(content=m["content"]) if m["role"]=="user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages[:-1]
            ]
            
            response = chain.invoke({"input": query_text, "chat_history": history})
            answer = response["answer"]
            sources = list(set(doc.metadata.get("source", "") for doc in response["context"]))
            
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources: st.write(f"- {s}")
            
            # Save AI response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            })

# ==========================================
# 3. UI LAYOUT
# ==========================================
with st.sidebar:
    st.title("🩺 Medline AI")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages:
    # This is the "Hero" section that fills the blank screen
    st.markdown("""
        <div style='text-align: center; padding: 40px 0 20px 0;'>
            <h1 style='color: #1976D2; font-size: 2.8rem; margin-bottom: 0;'>
                How can I help you today?
            </h1>
            <p style='color: #64748b; font-size: 1.2rem; margin-top: 10px;'>
                Search thousands of verified medical documents instantly.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    starters = ["What are symptoms of hypertension?", "How to prepare for a blood test?", "Explain Type 2 Diabetes", "Common side effects of Ibuprofen"]
    
    for i, text in enumerate(starters):
        if (col1 if i % 2 == 0 else col2).button(text, use_container_width=True):
            handle_query(text)
            st.rerun()

else:
    # Display Chat History
    for msg in st.session_state.messages:
        # Note: We skip the LAST user message if we are currently in handle_query 
        # to avoid double rendering, but handle_query ends with a rerun, 
        # so this loop handles everything upon refresh.
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources"):
                    for s in msg["sources"]: st.write(f"- {s}")

if user_input := st.chat_input("Type your question..."):
    handle_query(user_input)
    st.rerun()