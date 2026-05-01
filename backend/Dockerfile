# docker file for the backend service of the MedlinePlus Chatbot application, using FastAPI and a RAG chain for retrieval and generation

FROM python:3.10-slim
WORKDIR /app
ENV GIT_PYTHON_REFRESH=quiet
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*
    
COPY backend/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt
COPY backend/ .
EXPOSE 8000
CMD ["uvicorn", "retrieve_api:app", "--host", "0.0.0.0", "--port", "8000"]