import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in the python path
sys.path.append(str(Path(__file__).parent.parent))

import nltk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Download NLTK tokenizers if needed (used by BM25 if preprocessing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Paths for storing persisted database and BM25 corpus
PROJECT_DIR = Path(__file__).parent.parent
DB_DIR = PROJECT_DIR / "chroma_db"
BM25_CORPUS_PATH = PROJECT_DIR / "bm25_corpus.json"

SAMPLE_DOCUMENTS = [
    {
        "text": "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant snippets from external databases and feeding them into the LLM context. This grounds the model and reduces hallucinations.",
        "metadata": {"title": "Introduction to RAG", "category": "AI Architecture"}
    },
    {
        "text": "Dense retrieval uses deep learning models to project text into high-dimensional vector spaces. These vectors capture conceptual meaning and semantic context, allowing semantic matching.",
        "metadata": {"title": "Dense Embeddings", "category": "Vector Search"}
    },
    {
        "text": "Traditional keyword search (like BM25) matches exact terms between the query and documents. It relies on term frequency and inverse document frequency, making it perfect for exact IDs or jargon.",
        "metadata": {"title": "Understanding BM25 & Keyword Search", "category": "Information Retrieval"}
    },
    {
        "text": "Ensemble retrievers combine different search methods, like dense semantic search and sparse BM25. They merge lists using algorithms like Reciprocal Rank Fusion (RRF) for higher initial recall.",
        "metadata": {"title": "The Power of Hybrid Search", "category": "Search Systems"}
    },
    {
        "text": "Reranking uses cross-encoder models to evaluate the precise relevance of retrieved candidate documents. It processes the query and each document together, scoring them to order the most relevant text to the top.",
        "metadata": {"title": "Cross-Encoders & Reranking Explained", "category": "Re-ranking"}
    },
    {
        "text": "A standard database works by executing SQL queries on relational tables, rows, and columns. A database schema defines how index constraints work and how tables link together.",
        "metadata": {"title": "How Relational Databases Work", "category": "Databases"}
    },
    {
        "text": "High-dimensional vector indexing engines partition vector space using algorithms like HNSW (Hierarchical Navigable Small World). This enables rapid retrieval of nearest neighbors based on distance metrics.",
        "metadata": {"title": "Vector Databases & HNSW Indexing", "category": "Vector Search"}
    }
]

def run_ingestion():
    print("--- Starting Document Ingestion ---")
    
    # 1. Prepare LangChain Document objects
    documents = []
    for doc in SAMPLE_DOCUMENTS:
        documents.append(
            Document(page_content=doc["text"], metadata=doc["metadata"])
        )
    
    # 2. Setup local dense embeddings
    print("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Setup local Chroma database and persist documents
    print(f"Indexing documents into Chroma Vector DB (stored at: {DB_DIR})...")
    
    # If the database folder already exists, delete it for a clean start
    if DB_DIR.exists():
        import shutil
        shutil.rmtree(DB_DIR)
        
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    print(f"Chroma DB successfully indexed {len(documents)} documents.")
    
    # 4. Save documents to JSON for local BM25 recreation
    print(f"Saving corpus for BM25 retriever to {BM25_CORPUS_PATH}...")
    serializable_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)
