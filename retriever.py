
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from config.settings import Settings


llm = OllamaLLM(model=Settings().chat_model, temperature=0.9)
vectordb = Chroma(persist_directory="./db/chroma", embedding_function=OllamaEmbeddings(model=Settings().embedding_model))
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

system_prompt = """ Act as a Clinical AI Assistant powered by MedlinePlus data. Your sole purpose is to synthesize the provided context to accurately answer the user's medical query.

### 1. Strict Grounding Constraints
* You must rely **strictly and exclusively** on the information contained within the `<context>` tags. 
* Do not inject external medical training, general knowledge, or diagnostic opinions under any circumstances.
* If the provided context does not contain the answer, or only partially answers it, you must state: "I do not have enough information from the provided MedlinePlus data to fully answer this question." Do not attempt to guess.

### 2. Output Formatting & Style
* **Tone:** Objective, professional, and empathetic.
* **Accessibility:** Translate complex medical jargon into simple, plain English. 
* **Structure:** Use bullet points for symptoms, treatments, or steps to maximize readability. Keep the response concise.

### 3. Mandatory Safety Guardrail
* You must append the following disclaimer to the end of every response:
  *"Disclaimer: This information is sourced from MedlinePlus for educational purposes and does not constitute professional medical advice."*

### Input Data
<context>
{context}
</context>

<query>
{input}
</query> """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, qa_chain)

while True:
    query = input("Enter your medical query (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break
    response = rag_chain.invoke({"input": query})
    print("\nResponse:\n", response["answer"])
    print("\n" + "="*50 + "\n")
    print("\n--- Sources Used ---")
    
    # The context is a list of Documents. We can loop through them to grab the URL!
    used_urls = set()
    for doc in response["context"]:
        if "source" in doc.metadata:
            used_urls.add(doc.metadata["source"])
            
    for url in used_urls:
        print(f"- {url}")
        
    print("\n" + "="*50 + "\n")
