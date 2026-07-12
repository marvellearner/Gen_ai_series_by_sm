import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in the python path
sys.path.append(str(Path(__file__).parent.parent))

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

DEFAULT_ENCODER_PATH = Path(__file__).parent.parent / "bm25_encoder.json"

def get_hybrid_retriever(index_name: str, encoder_path: Path = DEFAULT_ENCODER_PATH) -> PineconeHybridSearchRetriever:
    """Loads the pre-fitted BM25 encoder and returns a configured PineconeHybridSearchRetriever."""
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"BM25 encoder not found at {encoder_path}. "
            "Please run the ingestion script (ingest.py) first to fit and save the encoder."
        )
    
    # 1. Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=api_key)
    
    # 2. Load the fitted BM25 Encoder
    print(f"Loading BM25 Encoder from {encoder_path}...")
    bm25_encoder = BM25Encoder()
    bm25_encoder.load(str(encoder_path))
    
    # 3. Setup Dense Embeddings
    print("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. Connect to Pinecone Index
    index = pc.Index(index_name)
    
    # 5. Initialize retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index
    )
    return retriever

def format_docs(docs) -> str:
    """Helper to format retrieved LangChain Documents into a context block."""
    formatted = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Untitled Document")
        category = doc.metadata.get("category", "General")
        formatted.append(f"[{i+1}] Source: {title} (Category: {category})\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def run_hybrid_rag_query(query: str, index_name: str, top_k: int = 3) -> dict:
    """Runs a Hybrid search RAG query, fetches context documents, and generates a response using LLM."""
    retriever = get_hybrid_retriever(index_name)
    
    # Configure top_k retrieve
    retriever.top_k = top_k
    
    # Retrieve documents to show source attribution
    print(f"Retrieving top {top_k} documents for query: '{query}'...")
    retrieved_docs = retriever.invoke(query)
    
    # Define RAG prompt template
    template = """You are an AI Assistant designed to answer questions based on the provided context.
Answer the question using only the context provided. If you don't know the answer or if the context doesn't contain the answer, say "I don't know based on the provided context." Keep the response informative, professional, and clear.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Setup LLM
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    
    # Chain components
    context_str = format_docs(retrieved_docs)
    
    chain = (
        {"context": lambda x: context_str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Generating response via LLM...")
    response = chain.invoke(query)
    
    return {
        "query": query,
        "answer": response,
        "source_documents": retrieved_docs
    }

if __name__ == "__main__":
    index_name = os.getenv("PINECONE_INDEX_NAME", "hybrid-search-rag")
    test_query = "What is Retrieval-Augmented Generation and why do we use Pinecone?"
    
    try:
        result = run_hybrid_rag_query(test_query, index_name)
        print("\n=== QUERY RESULT ===")
        print(f"Query: {result['query']}")
        print(f"Answer:\n{result['answer']}")
        print("\n=== SOURCES CITED ===")
        for i, doc in enumerate(result['source_documents']):
            print(f"[{i+1}] {doc.metadata.get('title')} (Score metadata if any: {doc.metadata})")
    except Exception as e:
        print(f"Error during query execution: {e}", file=sys.stderr)
        sys.exit(1)
