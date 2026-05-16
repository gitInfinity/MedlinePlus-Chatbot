from dataclasses import dataclass

@dataclass
class Settings:
    chat_model = "deepseek-v4-flash:cloud"
    embedding_model = "nomic-embed-text:latest"
    sitemap_url = "https://medlineplus.gov/sitemap.xml"
    persistent_directory = "./db/chroma"
    test_directory = "./db/test_chroma"
    mock_directory = "./db/scripts_chroma"