# RAG-Based Customer Support Assistant with LangGraph & HITL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq-green.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-blue.svg)](https://www.trychroma.com/)

An advanced, production-ready Customer Support Assistant that utilizes **Retrieval-Augmented Generation (RAG)** grounded in proprietary PDF documentation. The system features a **Human-in-the-Loop (HITL)** escalation mechanism orchestrated by **LangGraph** to ensure 100% accuracy for complex queries.

---

## 🚀 Key Features

- **Document Ingestion**: Seamlessly digests multi-page PDF documents into semantic chunks.
- **Semantic Search**: Uses `all-MiniLM-L6-v2` local embeddings for high-speed retrieval from **ChromaDB**.
- **State-of-the-Art LLM**: Powered by **Groq (Llama-3-70b)** for sub-second reasoning and generation.
- **Workflow Orchestration**: Built with **LangGraph** as a state machine to handle complex logic transitions.
- **Intelligent Escalation**: Automatically detects low-confidence answers or sensitive complaints and routes them to a human agent.
- **Citations**: Grounded responses that ensure the AI doesn't hallucinate.

---

## 🏗️ Architecture Overview

The system follows a modular architecture divided into four main phases:

1.  **Ingestion**: Raw PDFs $\rightarrow$ Text Extraction $\rightarrow$ Recursive Chunking $\rightarrow$ Vector Embedding.
2.  **Retrieval**: User Query $\rightarrow$ Semantic Search $\rightarrow$ Context Augmentation.
3.  **Generation**: Augmented Prompt $\rightarrow$ Groq LLM Inference $\rightarrow$ Response Validation.
4.  **Escalation**: Intent Detection $\rightarrow$ Conditional Routing $\rightarrow$ Human Handoff (if required).

> [!NOTE]
> For a deep dive into the system design, check the `docs/` folder for comprehensive **HLD**, **LLD**, and **Technical Documentation** in LaTeX and PDF formats.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Orchestration**: LangGraph / LangChain
- **LLM API**: Groq (Llama-3-70b)
- **Vector Database**: ChromaDB (Local Persistent)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **PDF Parsing**: PyPDFLoader

---

## 📂 Project Structure

```text
├── chroma_db/            # Persistent vector database storage
├── data/                 # Input PDFs (knowledge_base.pdf)
├── docs/                 # Detailed Engineering Documentation
│   ├── HLD           
│   ├── LLD
│   └── Technical Doc
├── src/                  # Core Source Code
│   ├── chunker.py        # Text splitting logic
│   ├── embeddings.py     # Local embedding model initialization
│   ├── llm.py            # Groq LLM configuration
│   ├── loader.py         # PDF loading utilities
│   ├── rag_pipeline.py   # LangGraph definition (nodes/edges)
│   └── retriever.py      # ChromaDB interaction layer
├── .env                  # Environment variables (API Keys)
├── main.py               # Application entry point (CLI)
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag-support-assistant.git
cd rag-support-assistant
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃 Usage

1.  Place your support documentation (PDF) in the `data/` folder and name it `knowledge_base.pdf`.
2.  Run the application:
    ```bash
    python main.py
    ```
3.  The system will:
    - Process the PDF and store embeddings in `chroma_db/` (first run only).
    - Initialize the LangGraph assistant.
    - Accept queries in the CLI.

---

## 📄 Documentation

This project includes extensive engineering documentation:
- **High-Level Design (HLD)**: Overall system architecture and data flow.
- **Low-Level Design (LLD)**: Module specifications and state transition logic.
- **Technical Documentation**: Setup guide, prompt engineering, and future roadmap.

All documents are located in `docs/`.

---

## 🛡️ Security
- **Local Data**: All document embeddings are stored locally in ChromaDB.
- **API Security**: Groq API calls are secured via `.env` configuration.
- **Safety First**: The HITL mechanism ensures that sensitive queries are never handled by the AI alone.
