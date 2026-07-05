import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in the python path
sys.path.append(str(Path(__file__).parent.parent))

import nltk
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Load environment variables
load_dotenv()

# Download NLTK tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Path to save BM25 encoder
DEFAULT_ENCODER_PATH = Path(__file__).parent.parent / "bm25_encoder.json"

# Sample documents for the demonstration
SAMPLE_DOCUMENTS = [
    {
        "text": "Retrieval-Augmented Generation (RAG) is a pattern that enhances LLMs with external knowledge bases. It fetches relevant document snippets and feeds them into the model context.",
        "metadata": {"title": "Introduction to RAG", "category": "AI Architecture"}
    },
    {
        "text": "Dense retrieval uses deep learning models like OpenAI's text-embedding-3-small to embed text into continuous high-dimensional vector spaces. It captures semantic meaning but can miss exact keywords.",
        "metadata": {"title": "Dense Embeddings Explained", "category": "Vector Search"}
    },
    {
        "text": "Sparse retrieval uses traditional keyword-matching algorithms like BM25. It creates high-dimensional, highly sparse vectors where each dimension represents a term frequency. It is excellent for exact match search.",
        "metadata": {"title": "Understanding BM25 & Sparse Search", "category": "Information Retrieval"}
    },
    {
        "text": "Hybrid search combines dense (semantic) and sparse (keyword) vectors. By calculating weighted scores from both dense distance metrics and sparse matching, hybrid search provides the best overall retrieval performance.",
        "metadata": {"title": "The Power of Hybrid Search", "category": "Search Systems"}
    },
    {
        "text": "Pinecone is a cloud-native vector database designed to store, manage, and query vector embeddings at scale. It has native support for hybrid search with dense-sparse index representations.",
        "metadata": {"title": "Pinecone Vector Database", "category": "Databases"}
    },
    {
        "text": "LangChain is a popular open-source framework designed to facilitate building applications using LLMs. It features components for chaining prompts, memory, vector stores, and retrievers.",
        "metadata": {"title": "LangChain Framework Overview", "category": "AI Orchestration"}
    }
]

def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set. Please check your .env file.")
    return Pinecone(api_key=api_key)

def prepare_pinecone_index(index_name: str, pc: Pinecone):
    """Creates a Pinecone Index if it doesn't already exist.
    Important: Hybrid search in Pinecone requires 'dotproduct' metric.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating Pinecone Index: '{index_name}'...")
        # OpenAI text-embedding-3-small and text-embedding-ada-002 use 1536 dimensions
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="dotproduct",  # Crucial: hybrid search requires dotproduct metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Default serverless region
            )
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Pinecone Index '{index_name}' already exists.")

def run_ingestion(index_name: str, encoder_path: Path = DEFAULT_ENCODER_PATH, documents: list = SAMPLE_DOCUMENTS):
    """Fits BM25 encoder, saves it, initializes index, and indexes documents into Pinecone."""
    pc = get_pinecone_client()
    prepare_pinecone_index(index_name, pc)
    
    # 1. Prepare texts and metadata
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    # 2. Setup Sparse Encoder (BM25)
    print("Fitting BM25 Encoder on document corpus...")
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(texts)
    
    print(f"Saving fitted BM25 Encoder to: {encoder_path}")
    bm25_encoder.dump(str(encoder_path))
    
    # 3. Setup Dense Embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    
    # 4. Connect to index
    index = pc.Index(index_name)
    
    # 5. Initialize LangChain Hybrid Retriever
    # The retriever acts as our interface for ingestion and retrieval
    print("Initializing PineconeHybridSearchRetriever...")
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index
    )
    
    # 6. Add texts to the index
    print(f"Upserting {len(texts)} documents to Pinecone (dense and sparse vectors)...")
    retriever.add_texts(texts=texts, metadatas=metadatas)
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    index_name = os.getenv("PINECONE_INDEX_NAME", "hybrid-search-rag")
    try:
        run_ingestion(index_name)
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)
