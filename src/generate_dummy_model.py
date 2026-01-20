import torch
from torchvision import models
from pathlib import Path

def create_dummy_vit_model(output_path):
    """Create a dummy ViT model for testing"""
    
    print("=" * 60)
    print("CREATING DUMMY VIT MODEL")
    print("=" * 60)
    
    print("\n1. Loading Vision Transformer...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    print("2. Modifying classification head...")
    model.heads.head = torch.nn.Linear(768, 2)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("3. Saving model...")
    torch.save(model.state_dict(), output_path)
    
    print(f"\n✓ Model saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")

if __name__ == "__main__":
    model_path = Path(r"F:\Projects\threat-hunting-agent\models\ids_vit.pth")
    create_dummy_vit_model(model_path)
