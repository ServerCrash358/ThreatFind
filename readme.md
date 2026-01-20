🛡️ Autonomous Threat Hunting Agent
Real‑time network threat detection system that combines a trained Vision Transformer (ViT) on CIC‑IDS 2017 with a Retrieval‑Augmented Generation (RAG) pipeline powered by Llama 3.1, orchestrated as an autonomous agent and visualized through a live dashboard.

🔍 Project Overview
This project implements an autonomous threat hunting agent that:

Listens for new network logs in a Firestore database.

Uses a Vision Transformer (ViT) trained on CIC‑IDS 2017 (tabular → image conversion) to detect anomalies.

Uses a RAG pipeline + Llama 3.1 to classify the threat and map it to MITRE ATT&CK techniques.

Automatically decides a response action (e.g., BLOCK_IP, NOTIFY_SOC, LOG_AND_MONITOR).

Updates a real‑time dashboard showing alerts, severity, MITRE IDs, confidence, and actions.

The system is designed as a production‑style pipeline suitable for demonstration, learning, and portfolio use.

🧩 Architecture
High‑level data flow:

Log Ingestion

New network log entries are written to the logs collection in Firestore (either via test scripts or future integrations).

Agent Monitor

A long‑running Python process subscribes to Firestore changes and detects new log documents.

For each new log, it invokes the ThreatHuntingAgent.

LangGraph Orchestrated Agent

The agent is a LangGraph‑based state machine with four main steps:

log_monitor – create an initial alert document.

analyze – run ViT anomaly detection and compute confidence.

classify – run RAG + LLM to map to MITRE ATT&CK and assign severity.

plan – choose response action based on severity.

ViT‑based Anomaly Detection

Network log features are converted to a grayscale image using an IGTD‑style mapping and resized to 
224
×
224
224×224.

A ViT‑B/16 model, fine‑tuned on CIC‑IDS 2017 image representations, outputs an attack probability used as the anomaly confidence.

RAG + LLM Threat Classification

A minimal MITRE ATT&CK corpus (JSON) is chunked and embedded into a FAISS vector store.

The log (and context) are used to retrieve the top‑K relevant MITRE techniques.

A Llama 3.1 model (via Ollama) is prompted with this context to generate:

Threat summary.

MITRE technique ID(s).

Severity level.

Response Planning

Simple rule‑based planner:

Critical → BLOCK_IP

High/Critical → NOTIFY_SOC

Otherwise → LOG_AND_MONITOR

The selected action and final status are written back to the alerts collection in Firestore.

Dashboard

A Flask dashboard reads from Firestore and displays:

Total alerts, counts by severity, and blocks.

A table of recent alerts with time, severity, MITRE ID, confidence, and recommended action.

Auto‑refreshes periodically for near real‑time visualization.

🛠️ Tech Stack
Core:

Python 3.11+

PyTorch, torchvision (ViT‑B/16)

LangChain (core, community, text splitters)

LangGraph (agent orchestration)

FAISS (vector search)

Ollama + Llama 3.1 (local LLM)

Data & Backend:

CIC‑IDS 2017 dataset (CSV)

Firebase / Firestore (NoSQL database)

Firebase Admin SDK (Python)

Frontend / Dashboard:

Flask

HTML/CSS (dark theme security dashboard)

📁 Project Structure
text
F:\Projects\threat-hunting-agent\
├── data\
│   ├── raw\
│   │   └── cicids2017\           # Original CIC‑IDS CSV files
│   ├── processed\
│   │   └── cleaned_cicids2017.csv
│   └── images\
│       └── cicids_images\        # Generated images (benign/attack)
├── models\
│   ├── ids_vit.pth               # Trained ViT model (~343 MB)
│   └── mitre_corpus.json         # MITRE ATT&CK subset
├── src\
│   ├── agent_state.py            # LangGraph state schema
│   ├── agent_orchestrator.py     # ThreatHuntingAgent (main agent)
│   ├── agent_monitor.py          # Firestore listener / runner
│   ├── firebase_client.py        # Firestore wrapper
│   ├── preprocess_data.py        # CIC‑IDS cleaning & merging
│   ├── generate_images.py        # Tabular → image transformation
│   ├── build_knowledge_base.py   # MITRE corpus builder
│   ├── rag_pipeline.py           # RAG + LLM classification
│   ├── train_vit_local.py        # ViT training script
│   └── test_agent.py             # Local end‑to‑end test harness
├── dashboard_flask.py            # Flask dashboard
├── inject_log.py                 # Test log injector
├── cleanup_logs.py               # Utility to clear Firestore collections
├── requirements.txt
├── serviceAccountKey.json        # Firebase service account (not committed)
└── README.md
📦 Installation & Setup
1. Clone the Repository
bash
git clone https://github.com/<your-username>/threat-hunting-agent.git
cd threat-hunting-agent
2. Create Virtual Environment & Install Dependencies
bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
3. Setup Firebase / Firestore
Create a Firebase project.

Enable Firestore in native mode.

Generate a service account key (Project Settings → Service Accounts → Generate key).

Save it as:

text
F:\Projects\threat-hunting-agent\serviceAccountKey.json
Ensure Firestore has the following collections (they will be auto‑created):

logs

alerts

🧪 Data Preparation
4. Download CIC‑IDS 2017
Download the CIC‑IDS 2017 dataset (CSV version) and place the files under:

text
data/raw/cicids2017/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
5. Preprocess the Dataset
bash
.\venv\Scripts\Activate.ps1
python src\preprocess_data.py
This will:

Load and merge all CSV files.

Clean NaNs, duplicates, single‑value columns.

Normalize labels to binary (0 = BENIGN, 1 = ATTACK).

Save data/processed/cleaned_cicids2017.csv.

6. Generate Images (Tabular → Image)
bash
python src\generate_images.py
This will:

Sample a balanced subset (e.g., 10k rows).

Convert each row to a grayscale image using an IGTD‑style method.

Save images into:

text
data/images/cicids_images/
    0_benign/
    1_attack/
🧠 Model Training (ViT)
7. Train Vision Transformer Locally
The ViT training script uses the generated images to fine‑tune ViT‑B/16 for binary classification (benign vs attack).

bash
python src\train_vit_local.py
Approximate training characteristics on a laptop with RTX 2050:

Batch size: 16

Epochs: 5

Time: ~2–3 hours

Achieved validation accuracy: ~93% (your run)

The best model is saved to:

text
models/ids_vit.pth
🧠 RAG + LLM (MITRE ATT&CK)
8. Build MITRE Knowledge Base
bash
python src\build_knowledge_base.py
This script:

Attempts to fetch MITRE ATT&CK techniques (or falls back to a minimal curated subset if full data is unavailable).

Normalizes them into a JSON list with id, name, description, tactics, and url.

Saves to:

text
models/mitre_corpus.json
9. RAG Pipeline
The RAG system (src/rag_pipeline.py):

Loads mitre_corpus.json.

Splits techniques into chunks (using LangChain text splitters).

Embeds them with Ollama embeddings.

Stores embeddings in a FAISS vector store.

At inference time:

Uses the log as a query.

Retrieves top‑K relevant techniques.

Prompts Llama 3.1 with the retrieved context plus the log.

Parses the LLM’s JSON output containing:

summary

mitre_id

severity

🤖 Agent Orchestration (LangGraph)
10. ThreatHuntingAgent
Defined in src/agent_orchestrator.py, the ThreatHuntingAgent:

Loads:

Firestore client (via firebase_client.py)

RAG system (rag_pipeline.py)

Trained ViT model (models/ids_vit.pth)

Defines a LangGraph workflow over an AgentState dict:

log_monitor → analyze → classify → plan.

Key behaviors:

_analyze

Converts the log to an image (same pipeline as training).

Runs ViT inference and computes attack probability.

Flags the log as anomaly if probability > 0.5.

Updates the corresponding alert document.

_classify

Calls the RAG pipeline to get:

Threat summary

MITRE ID

Severity

Updates the alert.

_plan

Decides response action from severity.

Updates alert with recommended_action and final status.

🛰️ Runtime Components
11. Start Ollama + Llama 3.1
Install Ollama and pull Llama 3.1 (once):

bash
ollama pull llama3.1:8b
Ensure the Ollama service is running:

bash
ollama serve
# or just run Ollama app on Windows and keep it open
12. Start Flask Dashboard
bash
.\venv\Scripts\Activate.ps1
python dashboard_flask.py
Visit:

text
http://localhost:5000
You should see the Autonomous Threat Hunting Agent dashboard, with zero alerts initially.

13. Start Agent Monitor
bash
.\venv\Scripts\Activate.ps1
python src\agent_monitor.py
This will:

Initialize the agent.

Connect to Firestore.

Start listening for new documents in logs.

You’ll see logs like:

text
✓ MONITORING ACTIVE
📊 Dashboard: http://localhost:5000
🔍 Watching: logs collection
14. Inject Test Logs
Use the included injector to create synthetic logs:

bash
.\venv\Scripts\Activate.ps1
python inject_log.py
This:

Writes several synthetic network logs (DDoS, Port Scan, SQLi, normal traffic) to Firestore.

Triggers the agent to process them in real time.

Watch:

Terminal (agent_monitor.py): step‑by‑step processing, ViT confidence, MITRE mapping, action.

Dashboard: new alerts appearing with severity, MITRE IDs, and actions.

🧹 Utilities
Clear Firestore Collections
If you want a clean start:

bash
python cleanup_logs.py
This script deletes all documents from logs and alerts.

📈 Results & Metrics
Dataset: CIC‑IDS 2017 (multiple days of mixed benign and attack traffic).

Model: ViT‑B/16 fine‑tuned on IGTD‑style network images.

Validation Accuracy: ~93% on the sampled balanced subset.

Inference Latency:

ViT: ~10–50 ms per sample on CPU.

RAG + LLM: ~1–2 seconds per sample on local Llama 3.1.

End‑to‑End Latency: Typically under 2–3 seconds from log insertion to alert.

🚧 Limitations & Future Work
Current MITRE corpus may be a reduced subset; can be extended to full ATT&CK.

Severity mapping is rule‑based; can be enhanced with a learned severity model.

ViT input uses a fixed subset of features; better feature engineering could improve accuracy.

Deployment is local + Firestore; can be containerized and deployed to Kubernetes or a full cloud stack.

Actions are advisory (e.g., BLOCK_IP); integrating with real firewalls/SIEM systems would complete the loop.