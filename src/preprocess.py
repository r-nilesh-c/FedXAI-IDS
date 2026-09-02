import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"
OUTPUT_FILE = "data/ciciot2023_curated_subset.csv"

# Configuration: (High-Level Category, Target Sample Quota)
# Setting sample quota to None retains 100% of the available rows.
FILE_CONFIG = {
    # Benign Baseline (~28% of total dataset)
    'BenignTraffic': ('Benign', 100000),
    
    # Web & Stealth Malware (Keep 100% of available rows)
    'Backdoor_Malware': ('Web_Malware', None),
    'BrowserHijacking': ('Web_Malware', None),
    'CommandInjection': ('Web_Malware', None),
    'SqlInjection': ('Web_Malware', None),
    'Uploading_Attack': ('Web_Malware', None),
    'XSS': ('Web_Malware', None),
    'DictionaryBruteForce': ('BruteForce', None),
    
    # Reconnaissance (~40,000 total)
    'Recon-PingSweep': ('Recon', None),
    'Recon-HostDiscovery': ('Recon', 9500),
    'Recon-OSScan': ('Recon', 9500),
    'Recon-PortScan': ('Recon', 9500),
    'VulnerabilityScan': ('Recon', 9500),
    
    # Spoofing (~40,000 total)
    'DNS_Spoofing': ('Spoofing', 20000),
    'MITM-ArpSpoofing': ('Spoofing', 20000),
    
    # DoS (~60,000 total)
    'DoS-HTTP_Flood': ('DoS', 15000),
    'DoS-SYN_Flood': ('DoS', 15000),
    'DoS-TCP_Flood': ('DoS', 15000),
    'DoS-UDP_Flood': ('DoS', 15000),
    
    # DDoS & Mirai Botnet (~82,500 total across 15 sub-types)
    'DDoS-ACK_Fragmentation': ('DDoS', 5500),
    'DDoS-HTTP_Flood-': ('DDoS', 5500),
    'DDoS-ICMP_Flood': ('DDoS', 5500),
    'DDoS-ICMP_Fragmentation': ('DDoS', 5500),
    'DDoS-PSHACK_Flood': ('DDoS', 5500),
    'DDoS-RSTFINFlood': ('DDoS', 5500),
    'DDoS-SlowLoris': ('DDoS', 5500),
    'DDoS-SynonymousIP_Flood': ('DDoS', 5500),
    'DDoS-SYN_Flood': ('DDoS', 5500),
    'DDoS-TCP_Flood': ('DDoS', 5500),
    'DDoS-UDP_Flood': ('DDoS', 5500),
    'DDoS-UDP_Fragmentation': ('DDoS', 5500),
    'Mirai-greeth_flood': ('DDoS', 5500),
    'Mirai-greip_flood': ('DDoS', 5500),
    'Mirai-udpplain': ('DDoS', 5500)
}

def preprocess_dataset(data_dir: str = DATA_DIR, output_file: str = OUTPUT_FILE):
    """
    Curates raw CICIoT2023 CSV files by mapping granular sub-folder attack types
    into 7 high-level parent categories, sampling rows according to configured quotas,
    sanitizing invalid numerical values, and encoding target labels.
    """
    processed_dfs = []
    print("Starting CICIoT2023 file aggregation and sub-class mapping...")
    
    for file_key, (high_level_class, sample_quota) in FILE_CONFIG.items():
        matching_files = glob.glob(os.path.join(data_dir, f"*{file_key}*.csv"))
        if not matching_files:
            print(f"[WARNING] Skipping missing raw file key: {file_key}")
            continue
            
        filepath = matching_files[0]
        df = pd.read_csv(filepath)
        
        # Sub-sample dataset if quota is specified
        if sample_quota is not None and len(df) > sample_quota:
            df = df.sample(n=sample_quota, random_state=42)
            
        # Map sub-folder attack to parent category
        df['parent_label'] = high_level_class
        
        # Remove raw sub-class label column to prevent duplicated target columns
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
            
        processed_dfs.append(df)
        print(f"Loaded {len(df):>7} rows from {os.path.basename(filepath)} -> Class: {high_level_class}")
        
    if not processed_dfs:
        raise FileNotFoundError(f"No matching CSV files found in directory: {data_dir}")
        
    # Combine into master dataframe
    master_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Clean non-finite floating point numbers and missing values
    print("Sanitizing infinite values and dropping nulls...")
    master_df = master_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Encode parent labels into numerical targets (0 to 6)
    label_encoder = LabelEncoder()
    master_df['label_encoded'] = label_encoder.fit_transform(master_df['parent_label'])
    
    # Build class mapping reference dict
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Label Encoding Map: {class_mapping}")
    
    # Save curated output CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    master_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Curated dataset saved to {output_file} | Shape: {master_df.shape}")
    
    return master_df, class_mapping

if __name__ == "__main__":
    preprocess_dataset()
