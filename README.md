# Germany RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) system that answers questions about German study, work, and residence regulations using official government documents only.

The project demonstrates how to build a grounded, citation-aware LLM application using modern AI tooling, clean backend architecture, and containerized deployment with Docker, with a clear path toward Kubernetes.

---

## Problem Statement

Information about German residence permits, student work regulations, job-seeker visas, and the EU Blue Card is:

- Distributed across multiple official sources
- Frequently misinterpreted or outdated online
- Risky to answer incorrectly using generic LLMs

This project addresses that by building an AI system that:

- Retrieves answers strictly from authoritative government documents
- Minimizes hallucinations via retrieval grounding
- Returns transparent citations (authority, document, page)

---

## Key Features

- Document-grounded RAG pipeline using real German government PDFs
- Semantic vector search with Sentence Transformers and Chroma
- Max Marginal Relevance (MMR) and similarity search
- FastAPI backend with typed request and response models
- Citation-aware LLM answers (no free-form hallucinations)
- Environment-based configuration (no hardcoded secrets)
- Dockerized deployment with persistent vector storage
- Makefile for reproducible local and containerized workflows

---

## System Architecture

```
User Question
    |
    v
FastAPI API (/query or /answer)
    |
    v
Vector Retrieval (Chroma DB)
    |
    v
Relevant Document Chunks
    |
    v
LLM (answer generation using retrieved context only)
    |
    v
Final Answer with Citations
```

---

## Tech Stack

Backend and APIs:
- Python 3.11
- FastAPI
- Pydantic

Retrieval and Machine Learning:
- LangChain
- Chroma (vector database)
- Sentence Transformers (all-MiniLM-L6-v2)
- Max Marginal Relevance (MMR)

LLM Integration:
- OpenAI API (configured via environment variables)
- Strict prompt grounding with citation enforcement

DevOps and Tooling:
- Docker
- Makefile
- .env-based configuration
- Git

---

## Getting Started

### Clone the Repository

```
git clone https://github.com/<your-username>/germany-rag-assistant.git
cd germany-rag-assistant
```

---

### Local Development (Without Docker)

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```
python -m uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

---

### Dockerized Run (Recommended)

Build the image:

```
make docker-build
```

Run the container:

```
make docker-run
```

This will:
- Load environment variables from .env
- Mount the local chroma_db directory
- Expose the API on port 8000

---

## Example Query

```
POST /answer

{
  "question": "When does the 18-month job-seeker residence permit start after graduation?",
  "category": "post-study",
  "use_mmr": true
}
```

The response includes:
- A concise, grounded answer
- Authoritative citations (authority, document, page)
- Only text retrieved from official sources

---

## Configuration and Secrets

Sensitive configuration such as API keys is never hardcoded.

Secrets are provided via:
- A .env file for local development
- Environment variables for Docker and future Kubernetes deployments

---

## Project Structure

```
.
├── app/
├── scripts/
├── infra/
│   └── docker/
├── chroma_db/
├── Makefile
├── requirements.txt
└── README.md
```

---

## Roadmap

- Kubernetes deployment (Deployment, Service, Secrets, PVC)
- CI/CD pipeline
- Automated grounding and citation evaluation
- Support for multiple LLM backends (local and cloud)

---

## Author

Youssef Mazen  
Applied AI / ML Engineering  
Focus on LLM systems, RAG, evaluation, and production deployment