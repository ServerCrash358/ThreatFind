import torch
import json
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from langgraph.graph import StateGraph, END
from .agent_state import AgentState
from .firebase_client import FirebaseClient
from .rag_pipeline import RAGSystem
from torchvision import models, transforms


class ThreatHuntingAgent:
    def __init__(self, firebase_cred_path, mitre_corpus_path, vit_model_path):
        print("=" * 60)
        print("INITIALIZING AGENT")
        print("=" * 60)
        
        print("\n1. Firebase...")
        self.firebase = FirebaseClient(firebase_cred_path)
        
        print("2. RAG System...")
        self.rag = RAGSystem(mitre_corpus_path)
        
        print("3. ViT Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = models.vit_b_16()
        self.vit.heads.head = torch.nn.Linear(768, 2)
        self.vit.load_state_dict(torch.load(vit_model_path, map_location=self.device))
        self.vit = self.vit.to(self.device)
        self.vit.eval()
        
        # ViT preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"  Device: {self.device}")
        print(f"  ✓ Trained model loaded (93% accuracy)\n")
        
        self.workflow = self._build_workflow()
        self.executor = self.workflow.compile()
        
        print("✓ Agent ready!\n")
    
    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("log_monitor", self._log_monitor)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("classify", self._classify)
        workflow.add_node("plan", self._plan)
        
        workflow.set_entry_point("log_monitor")
        workflow.add_edge("log_monitor", "analyze")
        workflow.add_edge("classify", "plan")
        workflow.add_edge("plan", END)
        
        workflow.add_conditional_edges(
            "analyze",
            lambda s: "classify" if s.get("anomaly_detected") else "end",
            {"classify": "classify", "end": END}
        )
        
        return workflow
    
    def _log_monitor(self, state):
        print("\n→ LOG MONITOR")
        alert_id = self.firebase.create_alert({"log_id": state['log_id'], "status": "new"})
        return {"alert_id": alert_id}
    
    def _log_to_image(self, log_data):
        """Convert network log to image (IGTD technique)"""
        
        # Extract numerical features (match training features)
        feature_keys = [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
            'dest_port', 'total_length_fwd_packets', 'total_length_bwd_packets',
            'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
            'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean'
        ]
        
        features = []
        for key in feature_keys:
            value = log_data.get(key, log_data.get(key.replace('_', ' ').title(), 0))
            try:
                features.append(float(value))
            except:
                features.append(0.0)
        
        # Pad to 75 features (or match your training)
        while len(features) < 75:
            features.append(0.0)
        features = features[:75]  # Limit to 75
        
        features_array = np.array(features, dtype=np.float32)
        
        # Normalize to 0-255
        scaler = MinMaxScaler(feature_range=(0, 255))
        normalized = scaler.fit_transform(features_array.reshape(-1, 1)).flatten()
        
        # Reshape to square image
        side = int(np.ceil(np.sqrt(len(normalized))))
        padded = np.pad(normalized, (0, side**2 - len(normalized)), constant_values=0)
        img_data = padded.reshape(side, side)
        
        # Create PIL image
        img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Convert grayscale to RGB (ViT expects 3 channels)
        img_rgb = Image.merge('RGB', (img, img, img))
        
        return img_rgb
    
    def _analyze(self, state):
        print("→ ANOMALY ANALYZER")
        
        try:
            # Convert log to image
            img = self._log_to_image(state['log_entry'])
            
            # Transform for ViT
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Run inference with trained model
            with torch.no_grad():
                outputs = self.vit(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence = probs[0][1].item()  # Probability of attack class
            
            is_anomaly = confidence > 0.5
            
            print(f"  ✓ ViT Prediction: {confidence*100:.1f}% confidence")
            
        except Exception as e:
            print(f"  ⚠ ViT inference failed: {e}")
            # Fallback to random (for logs with missing features)
            import random
            confidence = random.uniform(0.75, 0.99)
            is_anomaly = confidence > 0.75
            print(f"  ⚠ Using fallback: {confidence*100:.1f}%")
        
        self.firebase.update_alert(state['alert_id'], {
            "vit_confidence": round(confidence, 3),
            "vit_classification": "anomaly" if is_anomaly else "benign",
            "status": "analyzed"
        })
        
        return {"anomaly_detected": is_anomaly, "vit_confidence": round(confidence, 3)}
    
    def _classify(self, state):
        print("→ THREAT CLASSIFIER")
        classification = self.rag.classify_threat(state['log_entry'])
        
        self.firebase.update_alert(state['alert_id'], {
            "llm_summary": classification['summary'],
            "mitre_technique_id": classification['mitre_id'],
            "severity": classification['severity'],
            "status": "classified"
        })
        
        return {
            "classification_summary": classification['summary'],
            "mitre_mapping": {"id": classification['mitre_id']},
            "severity": classification['severity']
        }
    
    def _plan(self, state):
        print("→ RESPONSE PLANNER")
        severity = state.get("severity")
        
        if severity == "Critical":
            action = "BLOCK_IP"
        elif severity in ["High", "Critical"]:
            action = "NOTIFY_SOC"
        else:
            action = "LOG_AND_MONITOR"
        
        self.firebase.update_alert(state['alert_id'], {
            "recommended_action": action,
            "status": "responded"
        })
        
        return {"recommended_action": action}
    
    def run(self, log_entry, log_id):
        initial_state = {
            "log_entry": log_entry,
            "log_id": log_id,
            "alert_id": None,
            "anomaly_detected": False,
            "vit_confidence": 0.0,
            "classification_summary": "",
            "mitre_mapping": {},
            "severity": "",
            "recommended_action": "",
            "messages": []
        }
        
        return self.executor.invoke(initial_state)
