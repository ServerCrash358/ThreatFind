"""Continuously monitor Firestore for new logs"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from firebase_admin import firestore
from src.agent_orchestrator import ThreatHuntingAgent

def monitor_logs():
    """Monitor Firestore and process new logs automatically"""
    
    print("\n" + "=" * 60)
    print("🛡️  AUTONOMOUS THREAT HUNTING AGENT - ACTIVE")
    print("=" * 60)
    
    # Initialize agent
    print("\n[1/3] Initializing agent...")
    agent = ThreatHuntingAgent(
        firebase_cred_path="serviceAccountKey.json",
        mitre_corpus_path="models/mitre_corpus.json",
        vit_model_path="models/ids_vit.pth"
    )
    
    print("\n[2/3] Connecting to Firestore...")
    db = firestore.client()
    
    # Track processed logs
    processed = set()
    
    print("\n[3/3] Starting real-time monitoring...")
    print("\n" + "=" * 60)
    print("✓ MONITORING ACTIVE")
    print("=" * 60)
    print(f"\n📊 Dashboard: http://localhost:5000")
    print(f"🔍 Watching: logs collection")
    print(f"⚡ Status: Ready to process threats\n")
    print("Press Ctrl+C to stop\n")
    
    # Callback for new documents
    def on_snapshot(doc_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == 'ADDED':
                doc = change.document
                log_id = doc.id
                
                # Skip if already processed
                if log_id in processed:
                    continue
                
                log_data = doc.to_dict()
                
                print(f"\n{'='*60}")
                print(f"🚨 NEW LOG DETECTED: {log_id}")
                print(f"{'='*60}")
                print(f"Source: {log_data.get('source_ip', 'N/A')}")
                print(f"Dest: {log_data.get('dest_ip', 'N/A')}:{log_data.get('dest_port', 'N/A')}")
                
                try:
                    result = agent.run(log_data, log_id)
                    
                    print(f"\n✓ ANALYSIS COMPLETE:")
                    print(f"  • Anomaly: {result.get('anomaly_detected')}")
                    print(f"  • Confidence: {result.get('vit_confidence')*100:.1f}%")
                    print(f"  • Severity: {result.get('severity')}")
                    print(f"  • MITRE: {result.get('mitre_mapping', {}).get('id', 'N/A')}")
                    print(f"  • Action: {result.get('recommended_action')}")
                    
                    processed.add(log_id)
                    
                except Exception as e:
                    print(f"\n✗ ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Watch logs collection
    logs_ref = db.collection('logs')
    logs_watch = logs_ref.on_snapshot(on_snapshot)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("✓ MONITORING STOPPED")
        print("=" * 60)
        logs_watch.unsubscribe()

if __name__ == "__main__":
    monitor_logs()
