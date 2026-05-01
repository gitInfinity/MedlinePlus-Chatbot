# retrive_api.py - FastAPI application for handling retrieval requests using a RAG chain

import os
import traceback

from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
import mlflow
from retriever import get_rag_chain
app = FastAPI(title = "MedlinePlus RAG API")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []
    
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").strip())
mlflow.set_experiment("Medline_Medical_RAG")
mlflow.openai.autolog()
mlflow.langchain.autolog()  
    
chain = get_rag_chain()

@app.post("/ask")
async def retrieve(query: QueryRequest):
    try:
        logging.info(f"Received Query: {query.query}")
        logging.info(f"Chat history length: {len(query.chat_history)}")
        
        response = chain.invoke({"input": query.query, "chat_history": query.chat_history})
        
        answer = response["answer"]
        sources = list(set(doc.metadata.get("source", "") for doc in response["context"]))
        
        logging.info(f"Generated answer: {answer[:100]}...")  # Log first 100 chars
        logging.info(f"Sources retrieved: {sources}")
        
        return {"answer": answer, "sources": sources}
    
    except TimeoutError as t:
        logging.error(f"Timeout during query: {query.query} Error: {str(t)}")
        raise HTTPException(status_code=408, detail="Request Timeout")
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"CRITICAL ERROR in /ask endpoint:\n{error_traceback}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {type(e).__name__} - {str(e)}"
        )
