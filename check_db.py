from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from backend.config.settings import Settings

# 1. Connect to the database
embeddings = OllamaEmbeddings(model=Settings().embedding_model)
vectorDB = Chroma(
    persist_directory="./db/chroma", 
    embedding_function=embeddings
)

# 2. Count the items
total_chunks = vectorDB._collection.count()

print(f"🎉 Your database currently holds {total_chunks} chunks of medical data!")