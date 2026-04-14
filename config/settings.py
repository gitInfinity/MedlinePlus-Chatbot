from dataclasses import dataclass

@dataclass
class Settings:
    chat_model = "llama3.2:3b"
    embedding_model = "nomic-embed-text:latest"
    sitemap_url = "https://medlineplus.gov/sitemap.xml"
    persistent_directory = "./db/chroma"