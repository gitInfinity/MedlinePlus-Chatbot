from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
import mlflow
from retriever import get_rag_chain
app = FastAPI(title = "MedlinePlus RAG API")

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

@app.post("/ask")
async def retrieve(query: QueryRequest):
    with mlflow.start_run(run_name="RAG Retrieval API Call"):
        try:
            logging.info("Received retrieval request")
            logging.info(f"Query: {query.query}")
            logging.info(f"Chat history length: {len(query.chat_history)}")
            
        
            chain = get_rag_chain()
            response = chain.invoke({"input": query.query, "chat_history": query.chat_history})
            
            answer = response["answer"]
            sources = list(set(doc.metadata.get("source", "") for doc in response["context"]))
            
            mlflow.log_param("query_length", len(query.query))
            mlflow.log_param("chat_history_length", len(query.chat_history))
            
            logging.info(f"Generated answer: {answer[:100]}...")  # Log first 100 chars
            logging.info(f"Sources retrieved: {sources}")
            
            return {"answer": answer, "sources": sources}
        
        except Exception as e:
            logging.error(f"Error during retrieval: {str(e)}")
            mlflow.log_param("server error", str(e))
            raise HTTPException(status_code=500, detail="Internal Server Error")
        
        except TimeoutError as t:
            logging.error(f"Timeout during query: {query.query} Error: {str(t)}")
            mlflow.log_param("timeout error", str(t))
            raise HTTPException(status_code=408, detail="Request Timeout")