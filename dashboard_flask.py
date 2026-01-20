from flask import Flask, render_template_string
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Threat Hunting Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px; color: white; }
        .stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat { background: #1e293b; padding: 20px; border-radius: 8px; text-align: center; border: 2px solid #3b82f6; }
        .stat h3 { font-size: 0.8rem; color: #94a3b8; margin-bottom: 10px; }
        .stat p { font-size: 2rem; font-weight: bold; }
        table { width: 100%; background: #1e293b; border-collapse: collapse; border-radius: 8px; overflow: hidden; }
        th { background: #334155; padding: 12px; text-align: left; color: #cbd5e1; font-weight: 600; }
        td { padding: 12px; border-bottom: 1px solid #334155; }
        tr:hover { background: #334155; }
        .severity-critical { color: #ef4444; font-weight: bold; }
        .severity-high { color: #f97316; font-weight: bold; }
        .severity-medium { color: #eab308; font-weight: bold; }
        .severity-low { color: #22c55e; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Autonomous Threat Hunting Agent</h1>
            <p>Real-time Network Security Monitoring</p>
        </div>
        
        <div class="stats">
            <div class="stat"><h3>Total Alerts</h3><p>{{ stats.total }}</p></div>
            <div class="stat"><h3>Critical</h3><p>{{ stats.critical }}</p></div>
            <div class="stat"><h3>High</h3><p>{{ stats.high }}</p></div>
            <div class="stat"><h3>Medium</h3><p>{{ stats.medium }}</p></div>
            <div class="stat"><h3>Low</h3><p>{{ stats.low }}</p></div>
            <div class="stat"><h3>Blocks</h3><p>{{ stats.blocks }}</p></div>
        </div>
        
        <h2>Recent Alerts</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Status</th>
                    <th>Severity</th>
                    <th>Summary</th>
                    <th>MITRE ID</th>
                    <th>Confidence</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {% for alert in alerts %}
                <tr>
                    <td>{{ alert.created_at }}</td>
                    <td>{{ alert.status }}</td>
                    <td class="severity-{{ alert.severity|lower }}">{{ alert.severity or 'N/A' }}</td>
                    <td>{{ alert.llm_summary or 'Processing...' }}</td>
                    <td>{{ alert.mitre_technique_id or '-' }}</td>
                    <td>{{ "%.1f%%"|format(alert.vit_confidence * 100) if alert.vit_confidence else '-' }}</td>
                    <td><strong>{{ alert.recommended_action or '-' }}</strong></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <p style="text-align: center; margin-top: 20px; color: #94a3b8;">
            Auto-refreshes every 10 seconds | Last updated: {{ last_updated }}
        </p>
    </div>
    
    <script>
        setTimeout(() => location.reload(), 10000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    alerts = []
    docs = db.collection('alerts').order_by('created_at', direction='DESCENDING').limit(100).stream()
    
    for doc in docs:
        alert = doc.to_dict()
        if alert.get('created_at'):
            alert['created_at'] = alert['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        else:
            alert['created_at'] = 'N/A'
        alerts.append(alert)
    
    stats = {
        'total': len(alerts),
        'critical': len([a for a in alerts if a.get('severity') == 'Critical']),
        'high': len([a for a in alerts if a.get('severity') == 'High']),
        'medium': len([a for a in alerts if a.get('severity') == 'Medium']),
        'low': len([a for a in alerts if a.get('severity') == 'Low']),
        'blocks': len([a for a in alerts if a.get('recommended_action') == 'BLOCK_IP'])
    }
    
    return render_template_string(
        HTML_TEMPLATE,
        alerts=alerts,
        stats=stats,
        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

if __name__ == "__main__":
    print("Dashboard running on: http://localhost:5000")
    app.run(debug=True, port=5000)
