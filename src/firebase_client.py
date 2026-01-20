import firebase_admin
from firebase_admin import credentials, firestore

class FirebaseClient:
    def __init__(self, credential_path):
        if not firebase_admin._apps:
            cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
        print("✓ Firebase initialized")
    
    def add_log(self, log_data):
        try:
            log_data['timestamp'] = firestore.SERVER_TIMESTAMP
            _, doc_ref = self.db.collection('logs').add(log_data)
            print(f"✓ Added log: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def create_alert(self, alert_data):
        try:
            alert_data['created_at'] = firestore.SERVER_TIMESTAMP
            _, doc_ref = self.db.collection('alerts').add(alert_data)
            print(f"✓ Created alert: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def update_alert(self, alert_id, alert_data):
        try:
            self.db.collection('alerts').document(alert_id).update(alert_data)
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
