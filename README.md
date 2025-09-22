# PDF-RAG-FastAPI

## Overview
An intelligent Q&A system that uses a PDF document as a knowledge base. Built with a FastAPI backend, it utilizes `sentence-transformers` for embeddings, FAISS for efficient search, and Ollama for a powerful LLM.

## Features
* **Retrieval-Augmented Generation (RAG):** Answers questions by retrieving relevant information from a provided PDF document.
* **Vector Search:** Uses FAISS to perform fast and efficient semantic search on document chunks.
* **Local LLM Integration:** Leverages Ollama to use a local Large Language Model for generating answers, ensuring data privacy and offline functionality.
* **Modular Design:** Separates core RAG logic from the API implementation, making the codebase clean and maintainable.
* **Interactive API Documentation:** Automatically generates a comprehensive API documentation page (`/docs`) with Swagger UI, thanks to FastAPI.

## Technologies
* **Backend Framework:** FastAPI
* **LLM Runtime:** Ollama
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Store:** FAISS
* **PDF Processing:** PyMuPDF (`fitz`)
* **Server:** Uvicorn

## Prerequisites
Before you begin, ensure you have the following installed:
* Python 3.8 or higher
* Ollama: [https://ollama.com/](https://ollama.com/)

## Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/bakeryehyaa/PDF-RAG-FastAPI.git](https://github.com/bakeryehyaa/PDF-RAG-FastAPI.git)
cd PDF-RAG-FastAPI
