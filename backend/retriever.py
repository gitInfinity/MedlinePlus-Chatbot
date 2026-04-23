# engine.py
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to Python's system path
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config.settings import Settings

def get_rag_chain(db_directory=None):
    if db_directory is None:
        db_directory = Settings().persistent_directory
        
    # --- 1. CORE COMPONENTS ---
    # Using 0.2 for the "bedside manner" balance we discussed
    llm = ChatOllama(model=Settings().chat_model, temperature=0.2) 
    embeddings = OllamaEmbeddings(model=Settings().embedding_model)
    
    vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    
    # --- 2. ADVANCED RETRIEVAL (RE-RANKING) ---
    # We fetch a wide net (15 chunks) first
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 15})

    # Initialize the Cross-Encoder scoring model
    print("⏳ Initializing Re-Ranker...")
    hf_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Compress the 15 chunks down to the 5 highest-scoring ones
    compressor = CrossEncoderReranker(model=hf_model, top_n=5)
    
    # Wrap the base retriever with the re-ranker
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # --- 3. CONTEXTUALIZER (History-Aware) ---
    contextualize_q_system_prompt = """You are a search query generator. Given a chat history and the latest user question, formulate a standalone search query.
    
    CRITICAL RULES:
    1. DO NOT answer the question.
    2. DO NOT add conversational filler.
    3. ONLY output the exact raw text of the search query.
    4. If the question makes sense on its own, just return the user's exact question.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # The history_aware_retriever now uses our Reranked Retriever!
    history_aware_retriever = create_history_aware_retriever(
        llm, rerank_retriever, contextualize_q_prompt
    )

    # --- 4. MAIN ANSWER GENERATOR ---
    system_prompt = """Act as a Clinical AI Assistant powered by MedlinePlus data. Your sole purpose is to synthesize the provided context to accurately answer the user's medical query.

    ### 1. Strict Grounding Constraints
    * You must rely **strictly and exclusively** on the information contained within the `<context>` tags. 
    * Do not inject external medical training, general knowledge, or diagnostic opinions.
    * If the provided context does NOT contain relevant information to answer the question, you must output exactly: "I do not have enough information in my database to answer that." and STOP.
    * Do NOT use filler phrases like "Based on the context" or "However...".
    
    ### 2. Output Formatting & Style
    * **Tone:** Objective, professional, and empathetic.
    * **Accessibility:** Translate complex medical jargon into simple, plain English. 
    * **Structure:** Use bullet points for symptoms, treatments, or steps. Keep it concise.

    ### Input Data
    <context>
    {context}
    </context>"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Combine the history-aware reranked retriever and the QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    
    return rag_chain