# 🐳 Containerized Local LLM Grounding Engine

A dual-pipeline, Retrieval-Augmented Generation (RAG) system engineered to provide highly accurate, document-grounded answers from private policy documents. This project demonstrates proficiency in orchestrating a hybrid architecture combining cloud-based and self-hosted embedding models with Dockerized vector and data persistence services.

---

## 📘 Project Overview

This repository showcases a robust solution for querying private knowledge bases (specifically, Kerala Government policy documents).  
The core technical achievement is the implementation of **two parallel RAG pipelines**:

### **1. Cloud-Powered Pipeline**
- Uses **Google’s Gemini embedding model** for vector creation.

### **2. Self-Hosted Pipeline**
- Utilizes **Xenova/transformers** to run a lightweight local Sentence Transformer model (`all-MiniLM-L6-v2`), providing a cost-effective and low-latency alternative.

All essential services—vector store, database, and cache—are managed and persisted using **Docker** and **Docker Compose**.

---

## ⭐ Key Features

- **Dual-Embedding RAG**: Fully functional Cloud API and Local Embedding pipelines.
- **Local Embedding Generation**: Runs Xenova models locally—no external embedding API required.
- **Containerized Infrastructure**: ChromaDB, PostgreSQL, and Redis orchestrated via Docker Compose.
- **Vector Persistence**: ChromaDB uses Docker volumes to store embeddings.
- **Streaming API**: Chat responses are streamed using Next.js API routes.

---

## 🛠 Technology Stack

| Category         | Technology            | Purpose                                              |
|------------------|------------------------|------------------------------------------------------|
| RAG/AI           | Google Gemini          | Large Language Model for final generation           |
| Embeddings       | Xenova/transformers    | Local embedding model (`all-MiniLM-L6-v2`)          |
| Vector DB        | ChromaDB               | Vector store for persistent RAG indexing            |
| Framework        | Next.js, TypeScript    | API routes & server-side logic                      |
| Infrastructure   | Docker, Docker Compose | Containerization & orchestration                    |
| Data Persistence | PostgreSQL             | Storage for metadata & application state            |
| Caching          | Redis                  | In-memory caching & session management              |
| Tooling          | LangChain Splitters    | Document chunking                                   |

---

## 🚀 Getting Started

Follow these steps to set up and run the system locally.

### **Prerequisites**
- Node.js (LTS)
- Docker & Docker Compose
- Gemini API Key (`.env.local`)

---

### **1. Clone and Install Dependencies**

```bash
# Clone the repository
git clone https://github.com/premsgdev/containerized-Local-LLM-ingest-retrieve.git 
cd containerized-Local-LLM-ingest-retrieve

# Install required Node.js packages, including Xenova
npm install
