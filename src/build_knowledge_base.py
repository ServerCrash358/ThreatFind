import json
import os
from pathlib import Path

def build_mitre_corpus(output_path):
    """Build MITRE ATT&CK knowledge base"""
    
    print("=" * 60)
    print("BUILDING MITRE KNOWLEDGE BASE")
    print("=" * 60)
    
    # Download MITRE data
    try:
        from mitreattack.stix20 import MitreAttackData
        print("\n1. Downloading MITRE ATT&CK data...")
        mitre_data = MitreAttackData("enterprise-attack.json")
    except Exception as e:
        print(f"\n✗ MITRE download failed: {e}")
        print("Creating minimal corpus instead...")
        # Fallback: Create basic corpus
        corpus = [
            {
                "id": "T1071",
                "name": "Application Layer Protocol",
                "description": "Adversaries may communicate using application layer protocols to avoid detection",
                "tactics": ["Command and Control"],
                "url": "https://attack.mitre.org/techniques/T1071"
            },
            {
                "id": "T1190",
                "name": "Exploit Public-Facing Application",
                "description": "Adversaries may attempt to exploit public-facing applications",
                "tactics": ["Initial Access"],
                "url": "https://attack.mitre.org/techniques/T1190"
            },
            {
                "id": "T1566",
                "name": "Phishing",
                "description": "Adversaries may send phishing messages to gain access",
                "tactics": ["Initial Access"],
                "url": "https://attack.mitre.org/techniques/T1566"
            }
        ]
    else:
        print("\n2. Parsing techniques...")
        techniques = mitre_data.get_techniques(remove_revoked_deprecated=True)
        print(f"  Found {len(techniques)} techniques")
        
        corpus = []
        for idx, technique in enumerate(techniques):
            try:
                ext_refs = [r for r in technique.external_references if r.source_name == "mitre-attack"]
                if not ext_refs:
                    continue
                
                ext_ref = ext_refs[0]
                tactics = mitre_data.get_tactics_by_technique(technique.id)
                tactic_names = [t.name for t in tactics] if tactics else []
                
                doc = {
                    "id": ext_ref.external_id,
                    "name": technique.name,
                    "description": technique.description,
                    "tactics": tactic_names,
                    "url": ext_ref.url if hasattr(ext_ref, 'url') else f"https://attack.mitre.org/techniques/{ext_ref.external_id}"
                }
                corpus.append(doc)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(techniques)}")
            except:
                continue
    
    print(f"\n✓ Built corpus with {len(corpus)} entries")
    
    # Save corpus
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return corpus

if __name__ == "__main__":
    output = Path(r"F:\Projects\threat-hunting-agent\models\mitre_corpus.json")
    build_mitre_corpus(output)
