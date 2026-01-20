# ThreatFind — Autonomous Threat Hunting Agent

ThreatFind is a real-time autonomous network threat hunting system that combines Vision Transformers (ViT) for anomaly detection with a Retrieval-Augmented Generation (RAG) pipeline powered by Llama 3.1. The system is orchestrated as an agentic workflow and visualized through a live SOC-style dashboard.

---

## Key Capabilities

- Real-time network log ingestion using Firestore  
- Vision Transformer (ViT-B/16) trained on CIC-IDS 2017 (tabular → image)  
- Autonomous agent orchestration using LangGraph  
- RAG + LLM threat classification mapped to MITRE ATT&CK  
- Automated response planning (BLOCK / NOTIFY / MONITOR)  
- Live Flask-based security dashboard  

---

## System Architecture

Network Logs → Firestore (logs) → Agent Monitor → LangGraph Agent  
Analyze (ViT) → Classify (RAG + LLM) → Plan (Response)  
→ Firestore (alerts) → Live Dashboard

---

## Agent Workflow

- log_monitor: Initialize alert and metadata  
- analyze: Run ViT inference and compute anomaly confidence  
- classify: RAG + LLM to generate MITRE ID and severity  
- plan: Decide response action  

---

## ViT-Based Anomaly Detection

- Dataset: CIC-IDS 2017  
- Feature encoding: IGTD-style tabular-to-image mapping  
- Image size: 224×224 grayscale  
- Model: ViT-B/16 (fine-tuned)  
- Output: Attack probability (confidence score)  

---

## RAG + LLM Threat Classification

- MITRE ATT&CK techniques stored as JSON  
- Chunked and embedded into FAISS  
- Context retrieved per log  
- Llama 3.1 (via Ollama) generates:
  - Threat summary
  - MITRE technique ID(s)
  - Severity level

---

## Response Planning Logic

Critical → BLOCK_IP  
High / Critical → NOTIFY_SOC  
Low / Medium → LOG_AND_MONITOR  

---

## Dashboard

Flask-based SOC dashboard displaying:
- Total alerts
- Severity distribution
- Blocked events
- Recent alerts with timestamp, confidence, MITRE ID, severity, and action

---

## Tech Stack

Core:
- Python 3.11+
- PyTorch, torchvision
- Vision Transformer (ViT-B/16)
- LangChain, LangGraph
- FAISS
- Ollama + Llama 3.1

Backend:
- CIC-IDS 2017
- Firebase / Firestore
- Firebase Admin SDK

Frontend:
- Flask
- HTML / CSS (dark theme)

---

## Project Structure
```
ThreatFind/
├── data/
│   ├── raw/cicids2017/
│   ├── processed/
│   └── images/cicids_images/
├── models/
│   ├── ids_vit.pth
│   └── mitre_corpus.json
├── src/
│   ├── agent_state.py
│   ├── agent_orchestrator.py
│   ├── agent_monitor.py
│   ├── firebase_client.py
│   ├── preprocess_data.py
│   ├── generate_images.py
│   ├── build_knowledge_base.py
│   ├── rag_pipeline.py
│   ├── train_vit_local.py
│   └── test_agent.py
├── dashboard_flask.py
├── inject_log.py
├── cleanup_logs.py
├── requirements.txt
├── serviceAccountKey.json
└── README.md
```
---

## Installation
```bash
https://github.com/ServerCrash358/ThreatFind.git 
cd ThreatFind  

python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  
```
---

## Firebase Setup

- Create Firebase project
- Enable Firestore (Native Mode)
- Generate service account key
- Save as serviceAccountKey.json (not committed)

Collections:
- logs
- alerts

---

## Runtime
```bash
ollama pull llama3.1:8b  
ollama serve  

python dashboard_flask.py  
python src/agent_monitor.py  
python inject_log.py  
```
---

## Performance

ViT inference: 10–50 ms  
RAG + LLM: 1–2 seconds  
End-to-end latency: < 3 seconds  

---

## Future Work

- Full MITRE ATT&CK corpus ingestion
- Learned severity model
- Kubernetes/Docker deployment
- SOAR and firewall integrations

---

ThreatFind demonstrates how agentic AI, deep learning, and LLMs can be integrated into a modern autonomous cyber defense system.
License included.
