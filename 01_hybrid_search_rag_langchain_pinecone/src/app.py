import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in the python path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.ingest import run_ingestion
from src.query import run_hybrid_rag_query

# ANSI color codes for premium CLI interface
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    banner = f"""
{CYAN}======================================================================
{BOLD}🤖 GEN AI SERIES - PROJECT 01: HYBRID SEARCH RAG{RESET}
{CYAN}   Powered by LangChain, Pinecone Vector DB, and OpenAI
======================================================================{RESET}
    """
    print(banner)

def check_env():
    """Validates that necessary keys are configured before running."""
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    missing = []
    if not openai_key or "your_" in openai_key:
        missing.append("OPENAI_API_KEY")
    if not pinecone_key or "your_" in pinecone_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("PINECONE_INDEX_NAME")
        
    if missing:
        print(f"\n{RED}{BOLD}⚠️ CONFIGURATION WARNING{RESET}")
        print(f"The following environment variables are missing or default in your `.env` file:")
        for key in missing:
            print(f"  - {key}")
        print(f"\nPlease copy `.env.example` to `.env` inside Project 01 and supply valid keys before running.")
        return False
    return True

def main():
    print_banner()
    
    if not check_env():
        print(f"\n{YELLOW}Starting in 'Demo Check' mode. Please note, API operations will fail until keys are supplied.{RESET}\n")
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "hybrid-search-rag")
    
    while True:
        print(f"{BOLD}Main Menu:{RESET}")
        print(f"  [{GREEN}1{RESET}] Ingest Sample Documents into Pinecone (Fit BM25 & Upsert)")
        print(f"  [{GREEN}2{RESET}] Run Hybrid Search RAG Query")
        print(f"  [{GREEN}3{RESET}] Exit")
        
        choice = input(f"\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            print(f"\n{CYAN}--- Document Ingestion ---{RESET}")
            if not check_env():
                print(f"{RED}Cannot run ingestion without valid keys configured in `.env`.{RESET}\n")
                continue
            
            confirm = input("Confirm document ingestion? This will create/update your Pinecone Index. (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    run_ingestion(index_name)
                    print(f"\n{GREEN}Ingestion step complete!{RESET}\n")
                except Exception as e:
                    print(f"\n{RED}Ingestion failed: {e}{RESET}\n")
            else:
                print(f"{YELLOW}Ingestion cancelled.{RESET}\n")
                
        elif choice == '2':
            print(f"\n{CYAN}--- Hybrid Search RAG Query ---{RESET}")
            if not check_env():
                print(f"{RED}Cannot run queries without valid keys configured in `.env`.{RESET}\n")
                continue
            
            # Check if BM25 encoder has been saved
            encoder_file = Path(__file__).parent.parent / "bm25_encoder.json"
            if not encoder_file.exists():
                print(f"{RED}BM25 Encoder file not found! Please run option [1] (Ingestion) first to initialize.{RESET}\n")
                continue
            
            query = input(f"Enter your question for RAG: ").strip()
            if not query:
                print(f"{YELLOW}Question cannot be empty.{RESET}\n")
                continue
                
            try:
                result = run_hybrid_rag_query(query, index_name)
                print(f"\n{GREEN}{BOLD}=== ANSWER ==={RESET}")
                print(result['answer'])
                print(f"\n{CYAN}{BOLD}=== SOURCES CITED ==={RESET}")
                for i, doc in enumerate(result['source_documents']):
                    title = doc.metadata.get('title', 'Unknown')
                    category = doc.metadata.get('category', 'Unknown')
                    print(f"  [{i+1}] {title} (Category: {category})")
                print("\n")
            except Exception as e:
                print(f"\n{RED}Query failed: {e}{RESET}\n")
                
        elif choice == '3':
            print(f"\n{GREEN}Thank you for trying out the Hybrid Search RAG Project! Goodbye!{RESET}\n")
            break
        else:
            print(f"\n{RED}Invalid choice. Please select 1, 2, or 3.{RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Program interrupted by user. Exiting...{RESET}\n")
