# Project 01: Hybrid Search RAG with LangChain & Pinecone Vector DB

This project demonstrates **Hybrid Search Retrieval-Augmented Generation (RAG)**. Hybrid search combines:
1. **Dense Retrieval (Semantic Search)**: Captures conceptual and semantic meaning (using OpenAI's `text-embedding-3-small` or `text-embedding-ada-002`).
2. **Sparse Retrieval (Keyword Search)**: Captures exact keyword matching, jargon, and specific IDs (using BM25 via the `pinecone-text` package).

By combining both using a weighted scoring formula, the retrieval engine achieves significantly higher precision and recall than using either method alone.

---

## 🛠️ Tech Stack
- **Orchestration**: [LangChain](https://github.com/langchain-ai/langchain)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **Embeddings (Dense)**: OpenAI Embeddings
- **Embeddings (Sparse)**: BM25 Encoder (`pinecone-text`)
- **LLM**: ChatOpenAI (`gpt-4o-mini`)

---

## 🚀 Setup & Execution

### 1. Prerequisites
Make sure you have:
- An [OpenAI API Key](https://platform.openai.com/).
- A [Pinecone Account & API Key](https://www.pinecone.io/).
- Created a Pinecone Index:
  - **Index Name**: `hybrid-search-rag` (or configure in `.env`)
  - **Dimension**: `1536` (matching OpenAI's default embeddings)
  - **Metric**: `dotproduct` (required for Pinecone Hybrid search)

### 2. Environment Setup
From this directory (`01_hybrid_search_rag_langchain_pinecone`):

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Install packages
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

### 4. Running the Code
We have three scripts inside `src/`:
- **`ingest.py`**: Loads sample files, builds/saves the BM25 sparse encoder, and upserts dense and sparse vector representations to Pinecone.
- **`query.py`**: Executes hybrid retrieval and generates answers via LangChain.
- **`app.py`**: A CLI dashboard to trigger both ingestion and query processes.

To run the pipeline CLI:
```bash
python src/app.py
```

### 5. Jupyter Notebook Walkthrough
For a step-by-step interactive explanation of how dense and sparse vectors are calculated and combined, open and run the Jupyter notebook:
```bash
jupyter notebook hybrid_search_rag.ipynb
```
