# rag_core.py

from textwrap import wrap
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import fitz
from ollama import Client

# Global variables for the model and index
embedding_model = None
index = None
chunks = None
ollama_client = Client()

def initialize_rag_system(file_path):
    """
    Initializes the RAG system by loading the PDF, creating chunks,
    generating embeddings, and building the FAISS index.
    """
    global embedding_model, index, chunks

    # Load and process the PDF
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Split text into chunks
    chunk_size = 500
    chunks = wrap(text, chunk_size)
    print(f"Document split into {len(chunks)} chunks.")

    # Generate embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks], dtype="float32")
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    print("FAISS index created successfully.")

def search(query, top_k=3):
    """
    Finds the most relevant chunks for a given query.
    """
    if embedding_model is None or index is None:
        raise RuntimeError("RAG system not initialized. Call initialize_rag_system first.")
        
    query_emb = embedding_model.encode(query)
    query_emb = np.array([query_emb], dtype="float32")
    distances, indices = index.search(query_emb, top_k)
    return [(chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]

def answer_question(query):
    """
    Retrieves context and generates a response from the LLM.
    """
    relevant_chunks = search(query, top_k=3)
    context = "\n".join([chunk for chunk, _ in relevant_chunks])

    messages = [
    {
        "role": "system",
        "content": "You are an AI assistant that should only rely on the supplied context when answering."
    },
    {
        "role": "user",
        "content": f"""Rely on the context below to respond to the question.
Guidelines:
1. If the answer is not in the context, reply with "I don't know".
2. Keep your response short—no more than five sentences.
3. Use strictly the provided context.

Context:
{context}

Question: {query}

Answer:"""
    }
]


    response = ollama_client.chat(
        model="llama3",
        messages=messages
    )

    # Correct way to access the response content
    return response['message']['content']