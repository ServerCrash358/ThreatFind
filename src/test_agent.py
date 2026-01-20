import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_orchestrator import ThreatHuntingAgent
from src.firebase_client import FirebaseClient

def test_system():
    """Test the complete threat hunting system"""
    
    print("\n" + "=" * 60)
    print("TESTING AUTONOMOUS THREAT HUNTING AGENT")
    print("=" * 60)
    
    try:
        # Initialize agent
        print("\n1. Initializing agent...")
        agent = ThreatHuntingAgent(
            firebase_cred_path="serviceAccountKey.json",
            mitre_corpus_path="models/mitre_corpus.json",
            vit_model_path="models/ids_vit.pth"
        )
        
        # Test logs
        test_logs = [
            {
                "name": "Potential C2 Communication",
                "data": {
                    "source_ip": "10.0.0.5",
                    "dest_ip": "192.168.1.100",
                    "dest_port": 6667,
                    "protocol": "TCP",
                    "flow_duration": 5000,
                    "total_fwd_packets": 15,
                    "total_bwd_packets": 10
                }
            },
            {
                "name": "Suspicious SSH Activity",
                "data": {
                    "source_ip": "172.16.0.10",
                    "dest_ip": "10.10.10.1",
                    "dest_port": 22,
                    "protocol": "TCP",
                    "flow_duration": 120000,
                    "total_fwd_packets": 500,
                    "total_bwd_packets": 300
                }
            }
        ]
        
        # Run tests
        print("\n2. Running tests...")
        for test in test_logs:
            print(f"\n{'=' * 60}")
            print(f"Test: {test['name']}")
            print(f"{'=' * 60}")
            
            # Add log to Firebase
            log_id = agent.firebase.add_log(test['data'])
            
            # Run agent
            result = agent.run(test['data'], log_id)
            
            # Show results
            print(f"\nResults:")
            print(f"  Anomaly Detected: {result.get('anomaly_detected')}")
            print(f"  Confidence: {result.get('vit_confidence'):.2f}")
            print(f"  Severity: {result.get('severity')}")
            print(f"  Action: {result.get('recommended_action')}")
        
        print(f"\n{'=' * 60}")
        print("✓ ALL TESTS COMPLETE!")
        print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
