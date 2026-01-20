from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    log_entry: dict
    log_id: str
    alert_id: Optional[str]
    anomaly_detected: bool
    vit_confidence: float
    classification_summary: str
    mitre_mapping: dict
    severity: str
    recommended_action: str
    messages: List[BaseMessage]
