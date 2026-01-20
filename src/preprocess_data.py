import pandas as pd
import numpy as np
from pathlib import Path

def clean_cicids_data(file_paths, output_path):
    print("=" * 60)
    print("STEP 1: LOADING CSV FILES")
    print("=" * 60)
    
    file_paths = [Path(f) for f in file_paths]
    df_list = []
    
    for file in file_paths:
        if not file.exists():
            print(f"⚠ File not found: {file}")
            continue
        
        print(f"Loading: {file.name}")
        try:
            df_temp = pd.read_csv(file, encoding='utf-8', low_memory=False)
            df_list.append(df_temp)
            print(f"  ✓ Loaded {len(df_temp):,} rows")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not df_list:
        raise ValueError("No CSV files loaded!")
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"\n✓ Total rows: {len(df):,}")
    
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING DATA")
    print("=" * 60)
    
    # Sanitize column names
    df.columns = df.columns.str.strip()
    
    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop NaN
    rows_before = len(df)
    df.dropna(inplace=True)
    print(f"✓ Dropped {rows_before - len(df):,} NaN rows")
    
    # Drop duplicates
    dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"✓ Dropped {dups:,} duplicates")
    
    # Create binary label
    if 'Label' in df.columns:
        print(f"\nLabel distribution:\n{df['Label'].value_counts()}")
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        print(f"\nBinary labels:\n{df['Label'].value_counts()}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return df

if __name__ == "__main__":
    data_dir = Path(r"F:\Projects\threat-hunting-agent\data\raw\cicids2017")
    
    # Updated file names (matching your directory)
    file_list = [
        data_dir / "Monday-WorkingHours.pcap_ISCX.csv",
        data_dir / "Tuesday-WorkingHours.pcap_ISCX.csv",
        data_dir / "Wednesday-workingHours.pcap_ISCX.csv",
        data_dir / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        data_dir / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        data_dir / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        data_dir / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",  # Note: DDos not DDoS
    ]
    
    output_file = Path(r"F:\Projects\threat-hunting-agent\data\processed\cleaned_cicids2017.csv")
    
    print("\n" + "=" * 60)
    print("CIC-IDS 2017 DATA PREPROCESSING")
    print("=" * 60 + "\n")
    
    df = clean_cicids_data(file_list, output_file)
    
    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 60)
