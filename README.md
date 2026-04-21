# MedlinePlus Local RAG Chatbot 🩺

A Retrieval-Augmented Generation (RAG) chatbot that queries medical information from MedlinePlus using local LLMs. All processing happens locally—no external API calls.

## Overview

This project builds a RAG-based chatbot that retrieves medical information from MedlinePlus documents and generates answers using a local LLM (Ollama). The system uses vector embeddings for semantic search and ChromaDB as a persistent vector store.

## Key Features

- **Local LLM Inference:** Uses Ollama to run LLMs locally (no API dependencies)
- **Semantic Search:** Vector embeddings enable semantic understanding of medical queries
- **Persistent Vector Store:** ChromaDB stores embeddings locally for fast retrieval
- **Source Attribution:** Responses include links to original MedlinePlus sources
- **Streamlit UI:** Simple web interface for querying

## Tech Stack

- **Language:** Python 3.10+
- **LLM:** Ollama (local inference)
- **Vector Database:** ChromaDB
- **Web Scraping:** BeautifulSoup, requests
- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **Embeddings:** Local embeddings or HuggingFace

## Project Structure

```
MedlinePlus-Chatbot/
├── src/
│   ├── app.py              # Streamlit application
│   ├── ingestion.py        # Web scraping and data processing
│   ├── retriever.py        # Vector search logic
│   └── __pycache__/        # Python cache
├── config/
│   └── settings.py         # Configuration settings
├── db/
│   ├── chroma/             # ChromaDB vector store
│   ├── scripts_chroma/     # Scripts for DB management
│   └── test_chroma/        # Test database
├── scripts/
│   └── mock_ingestion.ipynb # Notebook for testing ingestion
├── tests/
│   ├── smoke_test.py       # Basic smoke tests
│   └── __pycache__/
├── .streamlit/             # Streamlit config
├── .venv/                  # Python virtual environment
├── .gitignore
├── pyvenv.cfg
├── check_db.py             # Utility to check database
├── requirements.txt        # Python dependencies
└── retriever.cpython-313.pyc # Compiled Python module

```

## Getting Started

### Prerequisites

- Python 3.10+
- Ollama ([install here](https://ollama.ai))
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/gitInfinity/MedlinePlus-Chatbot.git
cd MedlinePlus-Chatbot
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Ollama** (in a separate terminal)
```bash
ollama serve
```

5. **Pull a model** (in another terminal)
```bash
ollama pull llama3      # Or: mistral, neural-chat, etc.
ollama list             # See available models
```

6. **Configure settings** (optional)
Edit `config/settings.py` to adjust:
- LLM model name
- ChromaDB path
- Chunk size and overlap
- Retrieval parameters

7. **Run the application**
```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

## Usage

1. Ask a medical question in the Streamlit interface
2. The system retrieves relevant MedlinePlus documents
3. The LLM generates an answer based on retrieved context
4. Response includes source links from MedlinePlus

## How It Works

### Data Ingestion
- Scrapes medical content from MedlinePlus using BeautifulSoup and requests
- Processes and chunks documents for optimal retrieval
- Stores chunks as vector embeddings in ChromaDB

### Query Processing
1. **Embedding:** User query is converted to vector embedding
2. **Retrieval:** Vector store returns most similar document chunks
3. **Generation:** LLM generates answer using retrieved context
4. **Attribution:** Links to source documents included in response

## Configuration

Key settings in `config/settings.py`:
```python
OLLAMA_MODEL = "llama3.2:3b"           # LLM model to use
CHROMA_DB_PATH = "./db/chroma"   # Vector store location
CHUNK_SIZE = 1000                 # Document chunk size
TOP_K = 5                        # Number of results to retrieve
```

## Testing

Run smoke tests to verify basic functionality:
```bash
python tests/smoke_test.py
```

Check database status:
```bash
python check_db.py
```

## Data Source

Medical information is sourced from MedlinePlus, a free service of the National Library of Medicine providing high-quality health information.

## Limitations

- **Not a substitute for professional medical advice:** Always consult a healthcare provider for medical concerns
- **Single-user only:** No authentication or multi-user support
- **Manual updates required:** Needs re-ingestion for new MedlinePlus content
- **English-only:** Supports English medical terminology
- **Depends on Ollama:** Requires local Ollama server running

## Technical Challenges

1. **Hardware-Aware Data Ingestion (The "8-Hour Marathon")** Ingesting 10,000 documents over 8 hours caused CPU heating and low-power mode throttling. Resolved this by implementing incremental batching and "cool-down" pauses to maintain stable processing speeds and prevent database corruption.
3. **Multi-Template HTML Parsing** MedlinePlus uses diverse layouts for encyclopedia, lab tests, and drug info. Developed a dynamic template dispatcher using BeautifulSoup to identify page types by URL and extract high-value text while ignoring navigation noise.
4. **Knowledge Leakage & Hallucinations:** During testing, the LLM attempted to use its own training data to answer medical history questions. Fixed this with Strict Context Guardrails—a system prompt that forces the model to rely only on retrieved chunks or admit ignorance.

## Requirements

See `requirements.txt` for full list. Core dependencies:
- langchain
- chromadb
- ollama
- streamlit
- beautifulsoup4
- requests

## License

This project is provided as-is for educational and personal use.

## Disclaimer

**Medical Information Notice:**
This tool is for informational purposes only and should not be used for diagnosis or treatment of medical conditions. Always consult with a qualified healthcare provider before making any medical decisions.

## Contact

- **GitHub:** [gitInfinity/MedlinePlus-Chatbot](https://github.com/gitInfinity/MedlinePlus-Chatbot)
- **Email:** rouhancyber123@gmail.com
