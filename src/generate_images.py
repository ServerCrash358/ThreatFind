import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import os

def tabular_to_image(data, img_size=(224, 224)):
    """Convert tabular data to image using IGTD technique"""
    # Normalize to 0-255
    scaler = MinMaxScaler(feature_range=(0, 255))
    normalized = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    
    # Reshape to square
    total_features = len(data)
    side = int(np.ceil(np.sqrt(total_features)))
    padded = np.pad(normalized, (0, side**2 - total_features), constant_values=0)
    img_data = padded.reshape(side, side)
    
    # Resize to target size
    img = Image.fromarray(img_data.astype(np.uint8), mode='L')
    img = img.resize(img_size, Image.LANCZOS)
    
    return img

def generate_images(csv_path, output_dir, sample_size=10000):
    """Generate images from CIC-IDS dataset"""
    
    print("=" * 60)
    print("GENERATING IMAGES FROM CICIDS")
    print("=" * 60)
    
    print(f"\n1. Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Sample data (for speed)
    if len(df) > sample_size:
        print(f"\n2. Sampling {sample_size:,} rows...")
        df_benign = df[df['Label'] == 0].sample(min(sample_size//2, len(df[df['Label'] == 0])), random_state=42)
        df_attack = df[df['Label'] == 1].sample(min(sample_size//2, len(df[df['Label'] == 1])), random_state=42)
        df = pd.concat([df_benign, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  ✓ Using {len(df):,} rows")
    print(f"  - Benign: {(df['Label'] == 0).sum():,}")
    print(f"  - Attack: {(df['Label'] == 1).sum():,}")
    
    # Separate features and labels
    labels = df['Label']
    features = df.drop('Label', axis=1)
    
    # Create output directories
    output_dir = Path(output_dir)
    benign_dir = output_dir / '0_benign'
    attack_dir = output_dir / '1_attack'
    benign_dir.mkdir(parents=True, exist_ok=True)
    attack_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n3. Generating images...")
    for idx, (_, row) in enumerate(features.iterrows()):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(features)}")
        
        # Convert to image
        img = tabular_to_image(row.values)
        
        # Save to appropriate folder
        label = labels.iloc[idx]
        if label == 0:
            img.save(benign_dir / f'benign_{idx}.png')
        else:
            img.save(attack_dir / f'attack_{idx}.png')
    
    print(f"\n✓ Generated {len(features):,} images")
    print(f"  ✓ Benign: {benign_dir}")
    print(f"  ✓ Attack: {attack_dir}")
    
    return output_dir

if __name__ == "__main__":
    csv_file = Path(r"F:\Projects\threat-hunting-agent\data\processed\cleaned_cicids2017.csv")
    images_dir = Path(r"F:\Projects\threat-hunting-agent\data\images\cicids_images")
    
    print("\n" + "=" * 60)
    print("IMAGE GENERATION")
    print("=" * 60 + "\n")
    
    generate_images(csv_file, images_dir, sample_size=10000)
    
    print("\n" + "=" * 60)
    print("✓ IMAGE GENERATION COMPLETE!")
    print("=" * 60)
