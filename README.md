# Gen AI Series: 20+ Python Projects 🚀

Welcome to the **Gen AI Series**! This repository is a curated collection of 20+ Generative AI, Large Language Model (LLM), and Retrieval-Augmented Generation (RAG) projects built using Python and modern AI frameworks.

Each project is self-contained in its own directory with its own dependencies and configuration, allowing you to run them independently without version conflicts.

---

## 🛠️ Project Roadmap

Below is the directory of all projects in this series, starting with Project 01.

| # | Project Name | Tech Stack | Status | Description |
|---|--------------|------------|--------|-------------|
| **01** | **[Hybrid Search RAG](file:///c:/Users/sarth/Gen_ai_series_by_sm/01_hybrid_search_rag_langchain_pinecone)** | LangChain, Pinecone, OpenAI, BM25 | ⚙️ In Progress | RAG application using hybrid search (dense + sparse retrieval) for higher accuracy. |
| **02** | *Coming Soon* | - | ⏳ Planned | Next Gen AI project. |

---

## ⚙️ Getting Started

To run any of the projects, navigate to its directory, set up a virtual environment, and install the local requirements.

### Quick Start Example (Project 01):

1. **Navigate to the project directory:**
   ```bash
   cd 01_hybrid_search_rag_langchain_pinecone
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```

6. **Run the application:**
   ```bash
   python src/app.py
   ```

---

## 💡 Best Practices Implemented

- **Isolation**: Each project contains its own virtual environment settings and dependencies.
- **Environment Safety**: Configuration details are managed strictly via `.env` files.
- **Modular Code**: Logic is structured cleanly into data ingestion (`ingest.py`), query pipeline (`query.py`), and execution entry points (`app.py`).
- **Interactive Guides**: Interactive Jupyter Notebooks (`.ipynb`) are included for educational walkthroughs.
