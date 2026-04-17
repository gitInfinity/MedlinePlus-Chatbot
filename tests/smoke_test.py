import os
import shutil
import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import Settings
from src.retriever import get_rag_chain

# 1. TEST CONFIGURATION
TEST_PERSISTENT_DIRECTORY = "./db/test_chroma"
TEST_URLS = [
    "https://medlineplus.gov/shock.html",
    "https://medlineplus.gov/fever.html"
]

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

    blacklist = blacklist = [
    "references", "review date", "last reviewed", "reviewer", 
    "related topics", "find an expert", "patient handouts", 
    "clinical trials", "languages", "topic image"
]
    headers = {'User-Agent': 'RouhanMedicalProject/1.0'}
    all_processed_chunks = []

    print(f"🚀 Starting Universal Smoke Test on {len(TEST_URLS)} URLs...")

    for url in TEST_URLS:
        try:
            # STEP 1: Get the raw HTML ourselves to avoid WebBaseLoader bugs
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            # STEP 2: Find the 'main' div (The Universal Anchor)
            main_div = soup.find("div", class_="main")
            if not main_div:
                print(f"⚠️ Could not find 'main' content for {url}")
                continue

            cleaned_text = ""
            
            # STEP 3: Extract Summary (Handles both templates)
            summary = (main_div.find("div", id="ency_summary") or 
                       main_div.find("section", id="topsum_section"))
            if summary:
                cleaned_text += summary.get_text(separator=" ", strip=True) + "\n\n"

            # STEP 4: Extract and Filter Sections
            sections = main_div.find_all("section")
            for section in sections:
                if section.get('id') in ["topsum_section", "toc-section"]:
                    continue
                
                header = section.find(['h2', 'h3'])
                if header:
                    header_text = header.get_text().lower().strip()
                    if any(word in header_text for word in blacklist):
                        continue
                
                cleaned_text += section.get_text(separator=" ", strip=True) + "\n\n"

            # STEP 5: Convert to LangChain format and split
            if cleaned_text.strip():
                doc = Document(page_content=cleaned_text.strip(), metadata={"source": url})
                chunks = text_splitter.split_documents([doc])
                meaningful_chunks = [c for c in chunks if len(c.page_content) > 150]
                all_processed_chunks.extend(meaningful_chunks)
                print(f"✅ Processed {url}: Captured {len(meaningful_chunks)} high-quality chunks.")
            else:
                print(f"⚠️ Warning: No medical text extracted from {url}")

        except Exception as e:
            print(f"❌ Error testing {url}: {e}")

    if all_processed_chunks:
        vectorDB.add_documents(all_processed_chunks)
        print(f"\n✨ Smoke Test Complete! Data saved to {TEST_PERSISTENT_DIRECTORY}")

def verify_results():
    print("\n--- 🧠 TESTING THE BRAIN ---")
    try:
        chain = get_rag_chain(TEST_PERSISTENT_DIRECTORY)
        test_query = "Should I give fluids to a person experiencing a fever? Should I give fluids to someone in shock?"  # A question that should be answerable from the test data
        print(f"Querying: {test_query}")
        
        response = chain.invoke({"input": test_query, "chat_history": []})
        
        print("\nAI ANSWER:")
        print(response["answer"])
        
        print("\n🔍 CONTEXT RETRIEVED (The Signal):")
        if not response["context"]:
            print("❌ ZERO CONTEXT RETRIEVED. The database is empty or search failed.")
        for i, doc in enumerate(response["context"]):
            print(f"\n[Chunk {i} from {doc.metadata.get('source', 'Unknown')}]:")
            print(doc.page_content[:300] + "...")
    except Exception as e:
        print(f"❌ Verification Error: {e}")

if __name__ == "__main__":
    run_smoke_test()
    verify_results()