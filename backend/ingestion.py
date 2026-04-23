import os
import gc
import time
import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import Settings

# --- CONFIGURATION ---
SITE_URL = Settings().sitemap_url
PERSISTENT_DIRECTORY = Settings().persistent_directory
HEADERS = {'User-Agent': 'RouhanMedicalProject/1.0'}

# Metadata Blacklist to ensure only medical knowledge is stored
BLACKLIST = [
    "references", "review date", "last reviewed", "reviewer", 
    "related topics", "find an expert", "patient handouts", 
    "clinical trials", "languages", "topic image"
]

REJECTED_URLS = ["organizations", "spanish", "druginfo", "encyclopedia", "videos", "imagepages", "quiz", "languages",
                 "faq", "medwords"]

def scrape_website(url: str, batch_size: int = 25):
    print(f"📥 Fetching sitemap from {url}...")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch sitemap: {e}")
        return

    soup = BeautifulSoup(response.content, "xml")
    all_urls = [loc.text for loc in soup.find_all("loc")]
    filtered_urls = []
    
    for url in all_urls:
        for reject in REJECTED_URLS:
            if reject in url:
                break
        else:
            filtered_urls.append(url)
            
    all_urls = filtered_urls
    del filtered_urls
    print(f"🗺️ Found {len(all_urls)} URLs to process.")

    # Initialize Vector Store
    embeddings = OllamaEmbeddings(model=Settings().embedding_model)
    vectorDB = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSISTENT_DIRECTORY,
    )
    
    # Improved splitter for medical context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, # Increased overlap to catch warnings
        separators=["\n\n", "\n", ". ", " "]
    )

    for i in range(0, len(all_urls), batch_size):
        batch_urls = all_urls[i:i + batch_size]
        batch_chunks = []
        
        current_batch_num = (i // batch_size) + 1
        print(f"\n⚙️ Processing batch #{current_batch_num}...")
        batch_start_time = time.time()
        
        for current_url in batch_urls:
            try:
                # 1. Fetch HTML raw
                page_response = requests.get(current_url, headers=HEADERS, timeout=10)
                page_response.raise_for_status()
                page_soup = BeautifulSoup(page_response.content, "html.parser")
                                
                # 2. Target the main content area
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
                
                # 3. Handle Topic vs Encyclopedia templates
                summary = (main_div.find("div", id="ency_summary") or 
                       main_div.find("section", id="topsum_section") or
                       main_div.find("div", class_="summary"))
                if summary:
                    cleaned_text += summary.get_text(separator=" ", strip=True) + "\n\n"

                # 4. Filter Sections based on the Blacklist
                sections = main_div.find_all(["section", "div"], recursive=False)
                if not sections: 
                    paragraphs = main_div.find_all("p")
                    cleaned_text += " ".join([p.get_text(strip=True) for p in paragraphs])
                else:
                    for section in sections:
                        if section.get('id') in ["topsum_section", "toc-section"]:
                            continue
                        
                        header = section.find(['h2', 'h3'])
                        if header:
                            header_text = header.get_text().lower().strip()
                            if any(word in header_text for word in BLACKLIST):
                                continue # Skip non-medical metadata sections
                        
                        cleaned_text += section.get_text(separator=" ", strip=True) + "\n\n"

                # 5. Document Creation
                if cleaned_text.strip():
                    doc = Document(page_content=cleaned_text.strip(), metadata={"source": current_url})
                    chunks = text_splitter.split_documents([doc])
                    meaningful_chunks = [c for c in chunks if len(c.page_content) > 150]
                    batch_chunks.extend(meaningful_chunks)
                    print(f"✅ Processed {current_url}: Captured {len(meaningful_chunks)} high-quality chunks.")
                else:
                    print(f"⚠️ Warning: No medical text extracted from {current_url}")
                
                page_soup.decompose()
                
                # Polite scraping: Avoid getting blocked
                time.sleep(0.5)

            except Exception as e:
                print(f"⚠️ Error loading {current_url}: {e}")

        # 6. Commit Batch to Chroma
        try:
            if batch_chunks:
                vectorDB.add_documents(batch_chunks)
                print(f"✅ Batch #{current_batch_num} complete: Added {len(batch_chunks)} chunks.")
            else:
                print(f"⚠️ Batch #{current_batch_num} resulted in no valid chunks.")
        except Exception as e:
            print(f"❌ DB Ingestion Error on Batch #{current_batch_num}: {e}")
        
        batch_end_time = time.time()
        print(f"Time taken for Batch #{current_batch_num}: {batch_end_time - batch_start_time:.2f} seconds")
        gc.collect()

    print("\n🎉 Ingestion complete! The database is now ready for re-ranked queries.")

if __name__ == "__main__":
    db_path = PERSISTENT_DIRECTORY
    scrape_website(SITE_URL)
    # # Safety Check: Prevent accidental overwrites
    # if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
    #     print(f"🛑 SAFETY LOCK: The database already exists at {db_path}!")
    #     print("Delete the folder first if you want to perform a fresh scrape.")
    # else:
    #     print("🚀 Launching Ingestion Pipeline...")
    #     scrape_website(SITE_URL)