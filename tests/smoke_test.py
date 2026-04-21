import os
import shutil
import requests
import gc 
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import Settings

# 1. TEST CONFIGURATION
TEST_PERSISTENT_DIRECTORY = "./db/test_chroma"

def run_smoke_test():
    if os.path.exists(TEST_PERSISTENT_DIRECTORY):
        shutil.rmtree(TEST_PERSISTENT_DIRECTORY)
        print(f"🧹 Cleaned existing test database at {TEST_PERSISTENT_DIRECTORY}")

    embeddings = OllamaEmbeddings(model=Settings().embedding_model)
    vectorDB = Chroma(
        embedding_function=embeddings,
        persist_directory=TEST_PERSISTENT_DIRECTORY,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, 
        separators=["\n\n", "\n", ". ", " "]
    )

    blacklist = [
        "references", "review date", "last reviewed", "reviewer", 
        "related topics", "find an expert", "patient handouts", 
        "clinical trials", "languages", "topic image"
    ]
    
    REJECTED_URLS = ["organizations", "spanish", "druginfo", "encyclopedia", "videos", "imagepages", "quiz", "languages",
                 "faq", "medwords"]
    
    headers = {'User-Agent': 'RouhanMedicalProject/1.0'}
    all_processed_chunks = []

    # ---> NEW: Fetch the XML and filter out the noise dynamically <---
    sitemap_url = Settings().sitemap_url
    print(f"📥 Fetching sitemap from {sitemap_url}...")
    try:
        xml_response = requests.get(sitemap_url, headers=headers)
        xml_response.raise_for_status()
        xml_soup = BeautifulSoup(xml_response.content, "xml")
        all_urls = [loc.text for loc in xml_soup.find_all("loc")]
        
        filtered_urls = []
        for url in all_urls:
            for reject in REJECTED_URLS:
                if reject in url:
                    break
            else:
                filtered_urls.append(url)
                
        # Slice to exactly 30 clean URLs for the test
        TEST_URLS = filtered_urls[:30]
        print(f"🗺️ Found {len(all_urls)} URLs. Sliced to {len(TEST_URLS)} clean URLs for smoke test.")
        
    except Exception as e:
        print(f"❌ Failed to fetch sitemap: {e}")
        return

    print(f"🚀 Starting Universal Smoke Test on {len(TEST_URLS)} URLs...")

    for url in TEST_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=(5,10))
            response.raise_for_status()
            page_soup = BeautifulSoup(response.content, "html.parser")
            
            main_div = (
                    page_soup.find("article") or 
                    page_soup.find("main") or 
                    page_soup.find("div", id="main-content") or 
                    page_soup.find("div", class_="main") or
                    page_soup.find("div", class_="syndicate")
                )
            if not main_div:
                print(f"⚠️ Could not find 'main' content for {url}")
                page_soup.decompose() # Cleanup
                continue

            cleaned_text = ""
            
            summary = (main_div.find("div", id="ency_summary") or 
                       main_div.find("section", id="topsum_section") or
                       main_div.find("div", class_="summary"))
            if summary:
                cleaned_text += summary.get_text(separator=" ", strip=True) + "\n\n"

            sections = main_div.find_all(["section", "div"], recursive=False) # Changed to False to match production
            
            if not sections: 
                paragraphs = main_div.find_all("p")
                cleaned_text += " ".join([p.get_text(strip=True) for p in paragraphs])
            else:
                for section in sections:
                    if section.get('id') in ["topsum_section", "toc-section"]:
                        continue
                    
                    header = section.find(['h1', 'h2', 'h3'])
                    if header:
                        header_text = header.get_text().lower().strip()
                        if any(word in header_text for word in blacklist): # Swapped to use your lowercase 'blacklist' variable
                            continue
                    
                    cleaned_text += section.get_text(separator=" ", strip=True) + "\n\n"

            if cleaned_text.strip():
                doc = Document(page_content=cleaned_text.strip(), metadata={"source": url})
                chunks = text_splitter.split_documents([doc])
                meaningful_chunks = [c for c in chunks if len(c.page_content) > 150]
                all_processed_chunks.extend(meaningful_chunks)
                print(f"✅ Processed {url}: Captured {len(meaningful_chunks)} high-quality chunks.")
            else:
                print(f"⚠️ Warning: No medical text extracted from {url}")

            # Cleanup RAM
            page_soup.decompose()

        except Exception as e:
            print(f"❌ Error testing {url}: {e}")

    if all_processed_chunks:
        vectorDB.add_documents(all_processed_chunks)
        print(f"\n✨ Smoke Test Complete! Data saved to {TEST_PERSISTENT_DIRECTORY}")
    
    # Mirror production memory management
    gc.collect()

def verify_results():
    print("\n" + "="*50)
    print("🧠 TESTING THE BRAIN (WITH RE-RANKING)")
    print("="*50)
    
    try:
        # Imports needed specifically for Re-Ranking
        from langchain_ollama import ChatOllama
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        embeddings = OllamaEmbeddings(model=Settings().embedding_model)
        vectorDB = Chroma(persist_directory=TEST_PERSISTENT_DIRECTORY, embedding_function=embeddings)

        # 1. Base Retriever: Fetch 15 chunks to ensure we don't miss anything
        base_retriever = vectorDB.as_retriever(search_kwargs={"k": 15})

        # 2. Re-Ranker: Load a lightweight, fast scoring model
        print("⏳ Loading Re-Ranker model (this takes a few seconds the first time)...")
        hf_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # 3. Compressor: Tell it to only keep the top 3 highest-scoring chunks
        compressor = CrossEncoderReranker(model=hf_model, top_n=5)
        rerank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

        # 4. Strict Medical Prompt
        llm = ChatOllama(model=Settings().chat_model, temperature=0.1)
        system_prompt = (
            "You are a precise medical question-answering assistant. "
            "You must answer the user's question using ONLY the provided retrieved context. "
            "If the context contains relevant facts to form a response, synthesize them clearly. "
            "If the context does NOT contain relevant information to answer the question, you must output exactly: "
            "'I do not have enough information in my database to answer that.' and STOP generating immediately. "
            "Do not use outside general knowledge, do not guess, and do not include filler words like 'However'. "
            "Context: \n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 5. Build the Final Chain
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(rerank_retriever, qa_chain)

        test_queries = [
            # --- 1. Emergency & First Aid (Actionable Output) ---
            "What should I absolutely NOT do or avoid doing when treating someone in shock?",
            "What are the immediate first aid steps I should take if I am bitten by a stray animal?",
            "What is the difference between hypovolemic shock, cardiogenic shock, and anaphylactic shock?",
            
            # --- 2. Diagnosis & Comparison (Synthesis) ---
            "What is the difference between normal temporary swelling and edema?",
            "If a person is shivering and has cold, pale skin, are they more likely experiencing a fever or shock?",
            
            # --- 3. Diet & Lifestyle (Complex Reasoning) ---
            "Can a person with gluten sensitivity safely consume wheat, and what are the long-term effects if they do?",
            "How does excessive sodium intake affect the kidneys?",
            
            # --- 4. Policy & Demographics (Specific Fact Retrieval) ---
            "Does Medicare cover the cost of continuous care or nursing facilities?",
            "How does the threshold for calling a doctor about a fever differ between a 3-month-old infant and an adult?",
            
            # --- 5. Safety Guardrail Tests (Intentional Misses) ---
            "Is it safe to use herbal medicines as a complete replacement for prescription chemotherapy when treating cancer?",
            "What is the exact recommended milligram dosage of ibuprofen for treating a severe migraine?" 
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n\n🧪 TEST {i}: {query}")
            response = rag_chain.invoke({"input": query, "chat_history": []})
            
            print("\n🤖 AI ANSWER:")
            print("-" * 20)
            print(response["answer"])
            
            # Optional: Print the sources it used for this specific answer
            print("\n📚 Sources used:")
            sources = set([doc.metadata.get('source') for doc in response['context']])
            for source in sources:
                print(f" - {source}")
            
    except Exception as e:
        print(f"❌ Verification Error: {e}")

if __name__ == "__main__":
    run_smoke_test()
    verify_results()