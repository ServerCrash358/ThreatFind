import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from pathlib import Path
import time

def train_vit_model(data_dir, output_path, num_epochs=5, batch_size=16):
    """Train Vision Transformer on CIC-IDS images"""
    
    print("=" * 60)
    print("TRAINING VISION TRANSFORMER")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("\n2. Loading dataset...")
    dataset = ImageFolder(data_dir, transform=train_transforms)
    print(f"   ✓ Total images: {len(dataset):,}")
    print(f"   ✓ Classes: {dataset.classes}")
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_set.dataset.transform = val_transforms
    
    print(f"   ✓ Training: {train_size:,}")
    print(f"   ✓ Validation: {val_size:,}")
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)
    
    # Model
    print("\n3. Building model...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(768, 2)  # 2 classes: benign/attack
    model = model.to(device)
    print("   ✓ Vision Transformer B/16 loaded")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Training loop
    print(f"\n4. Training for {num_epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {100*train_correct/train_total:.2f}%")
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Results:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"    Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"    ✓ Best model saved ({val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\n{'='*60}")
    print(f"✓ TRAINING COMPLETE!")
    print(f"  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Model saved: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    data_dir = Path(r"F:\Projects\threat-hunting-agent\data\images\cicids_images")
    model_path = Path(r"F:\Projects\threat-hunting-agent\models\ids_vit.pth")
    
    print("\n" + "=" * 60)
    print("VISION TRANSFORMER TRAINING")
    print("=" * 60 + "\n")
    
    train_vit_model(data_dir, model_path, num_epochs=5, batch_size=16)
    
    print("\nNext: python src\\test_agent.py")
