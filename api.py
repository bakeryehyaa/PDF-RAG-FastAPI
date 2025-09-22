from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

# Import the core logic from your rag_core file
from rag import initialize_rag_system, answer_question

# Define a Pydantic model for the request body
class Query(BaseModel):
    query: str

# Create the FastAPI application 
app = FastAPI(title="RAG API", description="An API for the Retrieval-Augmented Generation system")

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# A startup event to initialize the RAG system when the server starts
@app.on_event("startup")
def startup_event():
    print("Starting up the RAG API...")
    pdf_file_path = "data/data.pdf"
    if not os.path.exists(pdf_file_path):
        print(f"Error: The file '{pdf_file_path}' was not found.")
        return
    
    initialize_rag_system(pdf_file_path)
    print("RAG system initialized and ready.")

# Define the API endpoint to handle the questions
@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = answer_question(query.query)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)