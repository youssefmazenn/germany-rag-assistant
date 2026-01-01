# 🇩🇪 Germany RAG Assistant

A **production-oriented Retrieval-Augmented Generation (RAG) system** for answering questions about **German immigration, study, and work regulations**, grounded exclusively in **official government documents**.

This is a **real backend system**, not a demo — featuring Docker, Kubernetes, persistent vector storage, autoscaling, and citation-safe answers.

---

## ✨ Key Features

- Document-grounded **RAG pipeline** using official German PDFs
- **Semantic vector search** with Sentence Transformers + ChromaDB
- **MMR (Max Marginal Relevance)** retrieval for diversity
- **FastAPI backend** with strict request/response contracts
- **Citation-aware answers** (no hallucinations)
- **Environment-based configuration** (no hardcoded secrets)
- **Dockerized deployment**
- **Kubernetes deployment** with:
  - Persistent volumes
  - Liveness & readiness probes
  - Resource requests & limits
  - Horizontal Pod Autoscaler (HPA)
- **Makefile** for reproducible workflows

---

## 🧠 System Architecture

User Question  
↓  
FastAPI API (`/query` or `/answer`)  
↓  
Vector Retrieval (ChromaDB)  
↓  
Relevant Document Chunks  
↓  
LLM (uses retrieved context only)  
↓  
Final Answer + Citations  

---

## 🛠 Tech Stack

### Backend
- Python 3.11
- FastAPI
- LangChain
- Sentence Transformers
- ChromaDB

### LLM
- OpenAI (`gpt-4o-mini` by default)

### Infra & DevOps
- Docker
- Kubernetes (Docker Desktop)
- PersistentVolumeClaims
- ConfigMaps & Secrets
- Metrics Server
- Horizontal Pod Autoscaler (HPA)
- Makefile

---

## 📁 Project Structure

```
germany-rag-assistant/
├── app/
│   └── main.py
├── scripts/
│   ├── ingest.py
│   └── hello_rag.py
├── data_raw/
│   └── pdfs/
├── chroma_db/
├── infra/
│   ├── docker/
│   │   └── Dockerfile
│   └── k8s/
│       ├── namespace.yaml
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── configmap.yaml
│       ├── pvc.yaml
│       └── hpa.yaml
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🚀 Kubernetes Deployment (Local)

This project supports **local Kubernetes deployment** (Docker Desktop Kubernetes) with a production-style setup.

### Prerequisites

- Docker Desktop (Kubernetes enabled)
- `kubectl` installed and configured
- Docker image built locally: `germany-rag-assistant:latest`

Verify cluster access:

```
kubectl get nodes
```

---

## 1️⃣ Build the Docker Image

```
make docker-build
```

---

## 2️⃣ Apply Kubernetes Manifests

```
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/pvc.yaml
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml
kubectl apply -f infra/k8s/hpa.yaml
```

Verify resources:

```
kubectl -n germany-rag get all
kubectl -n germany-rag get hpa
```

---

## 3️⃣ Create the OpenAI Secret (REQUIRED)

⚠️ **Never commit API keys to Git**

```
kubectl -n germany-rag delete secret rag-secrets --ignore-not-found
kubectl -n germany-rag create secret generic rag-secrets \
  --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY"
```

Restart the deployment:

```
kubectl -n germany-rag rollout restart deployment rag-api
```

---

## 4️⃣ Access the API Locally

Port-forward the service:

```
kubectl -n germany-rag port-forward svc/rag-api 8000:8000
```

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

---

## 5️⃣ Verify Metrics & Autoscaling

Metrics (metrics-server required):

```
kubectl top nodes
kubectl top pods -n germany-rag
```

Describe HPA:

```
kubectl -n germany-rag describe hpa rag-api-hpa
```

---

## 6️⃣ (Optional) Trigger Autoscaling

Generate sustained load:

```
while true; do
  curl -s -X POST http://127.0.0.1:8000/answer \
    -H "Content-Type: application/json" \
    -d '{"question":"What are the main requirements for the EU Blue Card in Germany?","use_mmr":true}' \
    > /dev/null
done
```

Observe scaling in another terminal:

```
kubectl -n germany-rag get pods -w
```

---

## ✅ Status

- Dockerized ✅  
- Kubernetes-ready ✅  
- Persistent storage ✅  
- Health probes ✅  
- Autoscaling ✅  
- Production-grade RAG ✅  

---

## 📌 Next Possible Extensions

- Ingress (NGINX / Traefik)
- HTTPS with cert-manager
- Cloud deployment (GKE / EKS)
- CI/CD pipeline
- Observability (Prometheus + Grafana)

---

**Author:** Youssef Mazen  
**Purpose:** Real-world AI + Backend + DevOps portfolio project