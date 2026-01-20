"""Manually inject test logs into Firestore"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.firebase_client import FirebaseClient

def inject_test_logs():
    """Add test logs to Firestore"""
    
    print("\n" + "=" * 60)
    print("LOG INJECTOR")
    print("=" * 60)
    
    firebase = FirebaseClient("serviceAccountKey.json")
    
    test_logs = [
        {
            "name": "DDoS Attack",
            "source_ip": "203.0.113.45",
            "dest_ip": "10.0.0.100",
            "dest_port": 80,
            "protocol": "TCP",
            "flow_duration": 100,
            "total_fwd_packets": 10000,
            "total_bwd_packets": 5
        },
        {
            "name": "Port Scan",
            "source_ip": "198.51.100.23",
            "dest_ip": "192.168.1.1",
            "dest_port": 443,
            "protocol": "TCP",
            "flow_duration": 5000,
            "total_fwd_packets": 100,
            "total_bwd_packets": 0
        },
        {
            "name": "SQL Injection Attempt",
            "source_ip": "192.0.2.77",
            "dest_ip": "10.10.10.5",
            "dest_port": 3306,
            "protocol": "TCP",
            "flow_duration": 3000,
            "total_fwd_packets": 50,
            "total_bwd_packets": 45
        },
        {
            "name": "Normal Web Traffic",
            "source_ip": "172.16.0.20",
            "dest_ip": "93.184.216.34",
            "dest_port": 443,
            "protocol": "TCP",
            "flow_duration": 2000,
            "total_fwd_packets": 10,
            "total_bwd_packets": 12
        }
    ]
    
    print("\nInjecting test logs...\n")
    
    for log in test_logs:
        log_id = firebase.add_log(log)
        print(f"✓ Injected: {log['name']} (ID: {log_id})")
    
    print(f"\n✓ {len(test_logs)} logs injected!")
    print("\nCheck your Agent Monitor window for real-time processing!")

if __name__ == "__main__":
    inject_test_logs()
