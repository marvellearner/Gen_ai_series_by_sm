import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in the python path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from flashrank import Ranker
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
DB_DIR = PROJECT_DIR / "chroma_db"
BM25_CORPUS_PATH = PROJECT_DIR / "bm25_corpus.json"

def get_ensemble_retriever(db_dir: Path = DB_DIR, corpus_path: Path = BM25_CORPUS_PATH) -> EnsembleRetriever:
    """Loads Chroma Vector DB and BM25 Corpus, returning a unified EnsembleRetriever."""
    if not db_dir.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"Required ingestion files not found. "
            "Please run the ingestion script (ingest.py) first to index documents."
        )
    
    # 1. Initialize local dense embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Load Chroma DB
    print(f"Loading Chroma Vector DB from {db_dir}...")
    db = Chroma(persist_directory=str(db_dir), embedding_function=embeddings)
    # Get top 5 dense candidates
    dense_retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # 3. Load BM25 Corpus and initialize sparse retriever
    print(f"Loading BM25 Corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in corpus
    ]
    # Get top 5 sparse candidates
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 5
    
    # 4. Combine into Ensemble Retriever (Reciprocal Rank Fusion)
    print("Combining retrievers into EnsembleRetriever (RRF)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def format_docs(docs) -> str:
    """Helper to format retrieved and re-ranked Documents into a context block."""
    formatted = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Untitled Document")
        category = doc.metadata.get("category", "General")
        formatted.append(f"[{i+1}] Source: {title} (Category: {category})\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def run_hybrid_rerank_rag_query(query: str, db_dir: Path = DB_DIR, corpus_path: Path = BM25_CORPUS_PATH, top_k: int = 3) -> dict:
    """Retrieves candidates via Hybrid search, re-ranks them using FlashRank, and runs the Gemini RAG chain."""
    
    # 1. Get base ensemble retriever (returns up to 10 total candidate docs)
    base_retriever = get_ensemble_retriever(db_dir, corpus_path)
    
    # 2. Set up FlashRank Reranker (loads ms-marco-MiniLM-L-12-v2 by default)
    print(f"Initializing FlashRank Reranker (extracting top {top_k} results)...")
    # We specify top_n=top_k to narrow down the context for the LLM
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_k)
    
    # 3. Create Contextual Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # 4. Retrieve and re-rank documents
    print(f"Executing hybrid retrieval and re-ranking for query: '{query}'...")
    reranked_docs = compression_retriever.invoke(query)
    
    # 5. Setup OpenAI LLM
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or "your_" in openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    
    # 6. Define prompt template
    template = """You are an AI Assistant designed to answer questions based on the provided context.
Answer the question using only the context provided. If you don't know the answer or if the context doesn't contain the answer, say "I don't know based on the provided context." Keep the response informative, professional, and clear.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 7. Execute Chain
    context_str = format_docs(reranked_docs)
    
    chain = (
        {"context": lambda x: context_str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Generating response via Gemini LLM...")
    response = chain.invoke(query)
    
    return {
        "query": query,
        "answer": response,
        "source_documents": reranked_docs
    }

if __name__ == "__main__":
    # Diagnostic test query
    test_query = "How do vector databases perform quick search?"
    try:
        result = run_hybrid_rerank_rag_query(test_query)
        print("\n=== QUERY RESULT ===")
        print(f"Query: {result['query']}")
        print(f"Answer:\n{result['answer']}")
        print("\n=== SOURCE DOCUMENTS (RE-RANKED TOP-N) ===")
        for i, doc in enumerate(result['source_documents']):
            title = doc.metadata.get('title')
            score = doc.metadata.get('relevance_score')
            print(f"[{i+1}] {title} (Relevance Score: {score:.4f})")
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)
