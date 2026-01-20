from firebase_functions import firestore_fn, options
from firebase_admin import initialize_app
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_orchestrator import ThreatHuntingAgent

initialize_app()
options.set_global_options(region="us-central1")

agent = None

def get_agent():
    global agent
    if agent is None:
        agent = ThreatHuntingAgent(
            "serviceAccountKey.json",
            "models/mitre_corpus.json",
            "models/ids_vit.pth"
        )
    return agent

@firestore_fn.on_document_created(document="logs/{logId}")
def process_new_log(event):
    log_data = event.data.to_dict()
    log_id = event.params["logId"]
    
    try:
        agent = get_agent()
        result = agent.run(log_data, log_id)
        print(f"✓ Processed {log_id}")
    except Exception as e:
        print(f"✗ Error: {e}")
