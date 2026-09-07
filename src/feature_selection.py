import os
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

INPUT_CSV = "data/ciciot2023_curated_subset.csv"
OUTPUT_CSV = "data/ciciot2023_bcoa_selected.csv"
SELECTED_FEATURES_JSON = "data/bcoa_selected_features.json"

class BinaryChimpOptimizer:
    """
    Binary Chimp Optimization Algorithm (BCOA) for Feature Selection in FedXAI-IDS.
    
    Mimics the hunting social hierarchy of chimps (Attacker alpha, Barrier beta,
    Chaser gamma, Driver delta) to search the combinatorial binary feature space
    and select a compact, high-precision subset of network traffic features.
    """
    def __init__(self, 
                 num_chimps: int = 12, 
                 max_iter: int = 10, 
                 w1: float = 0.95, 
                 w2: float = 0.05, 
                 seed: int = 42):
        self.num_chimps = num_chimps
        self.max_iter = max_iter
        self.w1 = w1  # Weight for classification error (1 - F1)
        self.w2 = w2  # Weight for feature ratio penalty (|selected| / total)
        self.seed = seed
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoidal transfer function mapping continuous velocity updates into probabilities [0, 1]."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))

    def _evaluate_fitness(self, 
                          binary_mask: np.ndarray, 
                          X_tr: np.ndarray, 
                          y_tr: np.ndarray, 
                          X_val: np.ndarray, 
                          y_val: np.ndarray) -> float:
        """
        Calculates fitness score for a candidate binary feature mask.
        Fitness = w1 * (1 - F1_score) + w2 * (num_selected_features / total_features)
        """
        num_selected = np.sum(binary_mask)
        total_features = len(binary_mask)
        
        # Penalize empty feature selections heavily
        if num_selected == 0:
            return 1.0
            
        # Select active feature columns
        active_indices = np.where(binary_mask == 1)[0]
        X_tr_sub = X_tr[:, active_indices]
        X_val_sub = X_val[:, active_indices]
        
        # Evaluate feature subset performance using DecisionTree
        clf = DecisionTreeClassifier(max_depth=6, random_state=self.seed)
        clf.fit(X_tr_sub, y_tr)
        y_pred = clf.predict(X_val_sub)
        
        score_f1 = f1_score(y_val, y_pred, average='weighted')
        error_rate = 1.0 - score_f1
        feature_ratio = num_selected / total_features
        
        fitness = (self.w1 * error_rate) + (self.w2 * feature_ratio)
        return fitness

    def optimize(self, df: pd.DataFrame, target_col: str = 'label_encoded', sample_size: int = 25000):
        """
        Executes BCOA feature selection on the input dataframe.
        """
        np.random.seed(self.seed)
        
        drop_cols = ['parent_label', 'label_encoded']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        total_features = len(feature_cols)
        
        print(f"\n[BCOA] Starting Binary Chimp Optimization across {total_features} features...")
        
        # Sample subset for fast metaheuristic evaluation
        if len(df) > sample_size:
            eval_df = df.sample(n=sample_size, random_state=self.seed)
        else:
            eval_df = df
            
        X = eval_df[feature_cols].values.astype(np.float32)
        y = eval_df[target_col].values.astype(np.int64)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=self.seed, stratify=y)
        
        # Initialize continuous velocities and binary positions for chimp population
        V = np.random.uniform(-1.0, 1.0, (self.num_chimps, total_features))
        X_bin = (self._sigmoid(V) > np.random.rand(self.num_chimps, total_features)).astype(int)
        
        # Social Hierarchy Leaders
        alpha_pos = np.ones(total_features, dtype=int)
        beta_pos = np.ones(total_features, dtype=int)
        gamma_pos = np.ones(total_features, dtype=int)
        delta_pos = np.ones(total_features, dtype=int)
        
        alpha_score = float('inf')
        beta_score = float('inf')
        gamma_score = float('inf')
        delta_score = float('inf')
        
        # Main Evolutionary Search Loop
        for iteration in range(1, self.max_iter + 1):
            # Dynamic decay parameter f decreasing linearly from 2.5 to 0
            f_decay = 2.5 - iteration * (2.5 / self.max_iter)
            
            # Evaluate population fitness
            for i in range(self.num_chimps):
                fitness = self._evaluate_fitness(X_bin[i], X_tr, y_tr, X_val, y_val)
                
                # Update social hierarchy (Alpha, Beta, Gamma, Delta)
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_pos = X_bin[i].copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta_pos = X_bin[i].copy()
                elif fitness < gamma_score:
                    gamma_score = fitness
                    gamma_pos = X_bin[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = X_bin[i].copy()
                    
            print(f"[BCOA Iter {iteration:>2}/{self.max_iter}] Best Fitness (Alpha): {alpha_score:.4f} | Selected Features: {np.sum(alpha_pos)}/{total_features}")
            
            # Update chimp positions according to leader influence
            for i in range(self.num_chimps):
                for d in range(total_features):
                    # Random coefficients for driving, barrier, chasing, attacking behaviors
                    r1, r2 = np.random.rand(), np.random.rand()
                    a1 = 2 * f_decay * r1 - f_decay
                    c1 = 2 * r2
                    
                    d_alpha = abs(c1 * alpha_pos[d] - X_bin[i, d])
                    x1 = alpha_pos[d] - a1 * d_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    a2 = 2 * f_decay * r1 - f_decay
                    c2 = 2 * r2
                    d_beta = abs(c2 * beta_pos[d] - X_bin[i, d])
                    x2 = beta_pos[d] - a2 * d_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    a3 = 2 * f_decay * r1 - f_decay
                    c3 = 2 * r2
                    d_gamma = abs(c3 * gamma_pos[d] - X_bin[i, d])
                    x3 = gamma_pos[d] - a3 * d_gamma
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    a4 = 2 * f_decay * r1 - f_decay
                    c4 = 2 * r2
                    d_delta = abs(c4 * delta_pos[d] - X_bin[i, d])
                    x4 = delta_pos[d] - a4 * d_delta
                    
                    # Compute continuous position update
                    V[i, d] = (x1 + x2 + x3 + x4) / 4.0
                    
                    # Convert to binary mask using sigmoid probability thresholding
                    if np.random.rand() < self._sigmoid(V[i, d]):
                        X_bin[i, d] = 1
                    else:
                        X_bin[i, d] = 0
                        
        selected_indices = np.where(alpha_pos == 1)[0]
        selected_feature_names = [feature_cols[idx] for idx in selected_indices]
        
        print(f"\n[BCOA COMPLETED] Selected {len(selected_feature_names)} features out of {total_features}.")
        print(f"Selected Features: {selected_feature_names}")
        
        return selected_feature_names, alpha_score, alpha_pos

def run_bcoa_feature_selection(input_csv: str = INPUT_CSV, 
                                output_csv: str = OUTPUT_CSV, 
                                json_out: str = SELECTED_FEATURES_JSON):
    """
    Loads curated dataset, runs BCOA feature selection, saves filtered CSV and feature JSON.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found. Run dataset curation first.")
        
    df = pd.read_csv(input_csv)
    bcoa = BinaryChimpOptimizer(num_chimps=12, max_iter=10, seed=42)
    selected_features, best_fitness, best_mask = bcoa.optimize(df)
    
    # Filter dataset columns to retain only selected features + metadata targets
    keep_cols = selected_features + ['parent_label', 'label_encoded']
    filtered_df = df[keep_cols]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    
    with open(json_out, 'w') as f:
        json.dump({
            "selected_features": selected_features,
            "num_selected": len(selected_features),
            "best_fitness": float(best_fitness)
        }, f, indent=4)
        
    print(f"[SUCCESS] Filtered dataset saved to {output_csv} | Saved metadata to {json_out}")
    return filtered_df, selected_features

if __name__ == "__main__":
    run_bcoa_feature_selection()

