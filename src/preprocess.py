import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

DEFAULT_CSV = "data/ciciot2023_bcoa_selected.csv"
FALLBACK_CSV = "data/ciciot2023_curated_subset.csv"

def load_and_partition_data(csv_path: str = None,
                            num_clients: int = 3,
                            alpha: float = 0.5,
                            batch_size: int = 128,
                            seed: int = 42):
    """
    Loads the curated/BCOA dataset, applies per-training preprocessing, 
    and partitions samples across edge clients using Dirichlet Dir(alpha) 
    for Non-IID label skew.
    
    Returns:
        client_dataloaders: Dict containing Train, Val, Test DataLoaders per client
        global_class_priors: Prior probabilities vector pi for Logit Adjustment Loss
        feature_cols: List of active feature names
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Resolve CSV file path
    if csv_path is None:
        if os.path.exists(DEFAULT_CSV):
            csv_path = DEFAULT_CSV
        elif os.path.exists(FALLBACK_CSV):
            csv_path = FALLBACK_CSV
        else:
            raise FileNotFoundError(f"Neither {DEFAULT_CSV} nor {FALLBACK_CSV} found.")
            
    print(f"\n[PREPROCESS] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    drop_cols = ['parent_label', 'label_encoded']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['label_encoded'].values.astype(np.int64)
    
    num_classes = len(np.unique(y))
    num_samples = len(y)
    feature_dim = X.shape[1]
    
    print(f"[PREPROCESS] Total Samples: {num_samples:,} | Feature Dim: {feature_dim} | Num Classes: {num_classes}")
    
    # Non-IID Dirichlet Partitioning across classes
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx = class_indices[c]
        np.random.shuffle(idx)
        
        # Sample proportions from Dirichlet distribution Dir(alpha)
        proportions = np.random.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, split_points)
        
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split)
            
    client_dataloaders = {}

    
    # Global class prior frequencies (pi)
    class_counts = np.bincount(y, minlength=num_classes)
    global_class_priors = class_counts / num_samples
    
    print("\n[PREPROCESS] Client Non-IID Label Distribution Summary:")
    for client_id in range(num_clients):
        c_idx = np.array(client_indices[client_id])
        np.random.shuffle(c_idx)
        
        # Train (80%) / Val (10%) / Test (10%) splits
        X_c, y_c = X[c_idx], y[c_idx]
        X_tr, X_temp, y_tr, y_temp = train_test_split(X_c, y_c, test_size=0.20, random_state=seed, stratify=y_c)
        X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
        
        # Client-specific class priors for Logit Adjustment
        c_class_counts = np.bincount(y_tr, minlength=num_classes)
        c_class_priors = (c_class_counts + 1e-5) / np.sum(c_class_counts + 1e-5)
        
        client_dataloaders[client_id] = {
            "train": DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True),
            "val": DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=batch_size, shuffle=False),
            "test": DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=batch_size, shuffle=False),
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "feature_names": feature_cols,
            "class_priors": c_class_priors,
            "sample_counts": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)}
        }
        
        print(f"  Client {client_id+1}: Train={len(y_tr):>6} | Val={len(y_va):>5} | Test={len(y_te):>5}")
        
    return client_dataloaders, global_class_priors, feature_cols

if __name__ == "__main__":
    load_and_partition_data()
