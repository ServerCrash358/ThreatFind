import React, { useEffect, useState } from 'react';
import { db } from './firebase-config';
import { collection, query, orderBy, limit, onSnapshot } from 'firebase/firestore';
import './App.css';

function App() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0, critical: 0, high: 0, medium: 0, low: 0, blocks: 0
  });

  useEffect(() => {
    const q = query(
      collection(db, 'alerts'),
      orderBy('created_at', 'desc'),
      limit(100)
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const data = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data(),
        created_at: doc.data().created_at?.toDate?.()?.toLocaleString() || 'N/A'
      }));

      setAlerts(data);
      
      const newStats = {
        total: data.length,
        critical: data.filter(a => a.severity === 'Critical').length,
        high: data.filter(a => a.severity === 'High').length,
        medium: data.filter(a => a.severity === 'Medium').length,
        low: data.filter(a => a.severity === 'Low').length,
        blocks: data.filter(a => a.recommended_action === 'BLOCK_IP').length
      };
      setStats(newStats);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const getSeverityColor = (severity) => {
    const colors = {
      'Critical': '#ef4444',
      'High': '#f97316',
      'Medium': '#eab308',
      'Low': '#22c55e'
    };
    return colors[severity] || '#64748b';
  };

  if (loading) return <div className="loading">Loading...</div>;

  return (
    <div className="App">
      <header className="header">
        <h1>🛡️ Autonomous Threat Hunting Agent</h1>
        <p>Real-time Network Security Monitoring</p>
      </header>

      <div className="stats">
        <div className="stat-card" style={{borderColor: '#3b82f6'}}>
          <h3>Total</h3>
          <p>{stats.total}</p>
        </div>
        <div className="stat-card" style={{borderColor: '#ef4444'}}>
          <h3>Critical</h3>
          <p>{stats.critical}</p>
        </div>
        <div className="stat-card" style={{borderColor: '#f97316'}}>
          <h3>High</h3>
          <p>{stats.high}</p>
        </div>
        <div className="stat-card" style={{borderColor: '#eab308'}}>
          <h3>Medium</h3>
          <p>{stats.medium}</p>
        </div>
        <div className="stat-card" style={{borderColor: '#22c55e'}}>
          <h3>Low</h3>
          <p>{stats.low}</p>
        </div>
        <div className="stat-card" style={{borderColor: '#8b5cf6'}}>
          <h3>Blocks</h3>
          <p>{stats.blocks}</p>
        </div>
      </div>

      <div className="alerts-container">
        <h2>Recent Alerts</h2>
        <table className="alerts-table">
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
            {alerts.map(alert => (
              <tr key={alert.id}>
                <td>{alert.created_at}</td>
                <td>{alert.status}</td>
                <td style={{color: getSeverityColor(alert.severity)}}>
                  <strong>{alert.severity || 'N/A'}</strong>
                </td>
                <td>{alert.llm_summary || 'Processing...'}</td>
                <td>{alert.mitre_technique_id || '-'}</td>
                <td>{alert.vit_confidence ? `${(alert.vit_confidence * 100).toFixed(1)}%` : '-'}</td>
                <td><strong>{alert.recommended_action || '-'}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
