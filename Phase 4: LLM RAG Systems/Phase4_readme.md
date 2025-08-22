# Phases 4: LLM/RAG Systems with Lahman Baseball Database

This repository is part of my **10-Project Data & AI Governance Roadmap**, demonstrating how governance, trust, and compliance come together across the AI lifecycle.  

- **Phase 4** focuses on **building retrieval-augmented generation (RAG) chatbots, integrating LLMs, and tracking observability metrics** to ensure trust and accuracy.  

---

## Phase 4: LLM / RAG Systems

### Project 7: RAG Chatbot Demo – Baseball Knowledge

- **Objective:** Build a RAG-based chatbot that answers questions about players, teams, and historical events  
- **Data Sources:** Lahman Baseball Database + PDFs and scraped baseball guides  
- **Approach:**  
  - Ingest documents into a **vector database** (FAISS / Weaviate)  
  - Connect LLM (OpenAI GPT API / AWS Bedrock) for retrieval and generation  
  - Include **source attribution** to prevent hallucinations  
- **Output:** Streamlit demo + GitHub repo + blog post explaining design and use  

### Project 8: LLM Observability & Monitoring – Baseball Queries

- **Objective:** Ensure operational trust in the chatbot  
- **Approach:**  
  - Track **query results, response accuracy, latency, and hallucinations**  
  - Build dashboards to visualize metrics and error patterns  
  - Log all interactions for auditability and governance  
- **Output:** Dashboard + Jupyter notebook + README documenting observability setup  

### Technologies Used

- **Python:** `pandas`, `langchain`  
- **Vector DB:** FAISS or Weaviate  
- **LLM:** OpenAI GPT API, AWS Bedrock, or alternative cloud LLMs  
- **Frontend / Demo:** Streamlit  
- **Document Parsing:** PyMuPDF, pdfplumber  
- **Monitoring & Logging:** AWS CloudWatch, MLflow  
- **Visualization:** Plotly, Matplotlib  

**Alternatives:** Azure OpenAI, Cohere, Anthropic LLMs; Pinecone, Milvus, ChromaDB; Prometheus + Grafana

### Strategic Alignment

- **Data Foundations:** Leveraged curated datasets from Phases 1–2  
- **Lineage & Observability:** Logged all query interactions, document sources, embeddings  
- **Security & Privacy:** Anonymized sensitive player information  
- **Trust as a Product Feature:** Dashboards visualize accuracy, latency, hallucinations  
- **Enterprise + Public Perspective:** AWS-native path (Bedrock + S3 + CloudWatch + MLflow) and Python/Streamlit path for public demo
