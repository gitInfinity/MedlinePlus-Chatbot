# engine.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from config.settings import Settings

def get_rag_chain(db_directory=None):
    if db_directory is None:
        db_directory = Settings().persistent_directory
    llm = ChatOllama(model=Settings().chat_model, temperature=0.3) 
    embeddings = OllamaEmbeddings(model=Settings().embedding_model)
    vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 1. The Contextualizer: Rewrites follow-up questions to make sense
    contextualize_q_system_prompt = """You are a search query generator. Given a chat history and the latest user question, formulate a standalone search query.
    
    CRITICAL RULES:
    1. DO NOT answer the question.
    2. DO NOT add conversational filler (e.g., "Here is the question", "The standalone query is", etc.).
    3. ONLY output the exact raw text of the search query.
    4. If the question makes sense on its own, just return the user's exact question and absolutely nothing else.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. The Main Answer Generator (Now accepts chat_history)
    system_prompt = """ Act as a Clinical AI Assistant powered by MedlinePlus data. Your sole purpose is to synthesize the provided context to accurately answer the user's medical query.

    ### 1. Strict Grounding Constraints
    * You must rely **strictly and exclusively** on the information contained within the `<context>` tags. 
    * Do not inject external medical training, general knowledge, or diagnostic opinions.
    * If the provided context contains the answer, provide it directly. Do NOT apologize, and do NOT use filler phrases like "Based on the context" or "However, I can provide..."    
    * Do not output the line "Based on the provided MedlinePlus data within the <context> tags" as user does not need to be reminded of this constraint.
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
        MessagesPlaceholder("chat_history"), # <--- Plugs the memory in here!
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Combine the history retriever and the QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    
    return rag_chain