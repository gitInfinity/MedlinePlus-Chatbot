import os
import time

os.environ["USER_AGENT"] = "RouhanMedicalProject/1.0"

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import Settings
import requests

SITE_URL = Settings().sitemap_url
PERSISTENT_DIRECTORY = Settings().persistent_directory
os.environ["USER_AGENT"] = "LocalRAGScraper/1.0"
def scrape_website(url: str, batch_size: int = 50):
    sitemap_url = url
    response = requests.get(sitemap_url)
    response.raise_for_status()  # Ensure we got a successful response
    soup = BeautifulSoup(response.content, "xml")
    main_content = soup.find_all("div", {"id": "main-content"})
    embeddings = OllamaEmbeddings(
    model=Settings().embedding_model,)
    vectorDB = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSISTENT_DIRECTORY,  # Where to save data locally, remove if not necessary
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "])
    all_urls = [loc.text for loc in soup.find_all("loc")]
    all_urls = all_urls[104:]
    for i in range(0, len(all_urls), batch_size):
        batch_urls = all_urls[i:i + batch_size]
        batch_chunks = []
        for current_url in batch_urls:
            try:
                loader = WebBaseLoader(current_url)
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                print(f"Processing batch #{i}")
                batch_chunks.extend(chunks)
                time.sleep(1)
                del docs
                del chunks
            except Exception as e:
                print(f"Error loading {current_url}: {e}")
        try:
            if batch_chunks:
                batch_chunks = [chunk for chunk in batch_chunks if len(chunk.page_content) > 200]
                if batch_chunks:
                    vectorDB.add_documents(batch_chunks)
                    print(f"✅ Successfully added {len(batch_chunks)} chunks to DB.")
                else:
                    print("⚠️ No chunks in this batch met the length requirement.")
        except Exception as e:
            print(f"An error occurred during ingestion: {e}")

        del batch_chunks
    print("All batches processed.")

if __name__ == "__main__":
    scrape_website(SITE_URL)
    db_path = PERSISTENT_DIRECTORY
    # Check if the folder exists AND has files inside it
    if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
        print(f"🛑 SAFETY LOCK ACTIVATED: The database already exists!,")
        print("If you really want to run this again, delete the './db/chroma' folder first.")
    else:
        print("Starting the ingestion pipeline...")
        scrape_website(SITE_URL)
    