from firebase_admin import credentials, firestore
import firebase_admin

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("Deleting old logs...")
logs = db.collection('logs').stream()
count = 0
for log in logs:
    log.reference.delete()
    count += 1

print(f"✓ Deleted {count} logs")

print("Deleting old alerts...")
alerts = db.collection('alerts').stream()
count = 0
for alert in alerts:
    alert.reference.delete()
    count += 1

print(f"✓ Deleted {count} alerts")
