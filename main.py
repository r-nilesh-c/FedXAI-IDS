import os
import json
import torch
import numpy as np
import flwr as fl

from src.dataset import FILE_CONFIG
from src.preprocess import load_and_partition_data
from src.feature_selection import run_bcoa_feature_selection
from src.client import create_client_fn
from src.server import TrimmedMeanStrategy
from src.model import IDS_MLP

GLOBAL_MODEL_PATH = "data/global_ids_model.pt"
METADATA_PATH = "data/fl_metadata.json"


def run_fl_simulation(num_rounds: int = 3, num_clients: int = 3, alpha: float = 0.5):
    """
    Executes the complete FedXAI-IDS Federated Learning pipeline:
    1. Dataset Curation (CICIoT2023 7 parent classes)
    2. Binary Chimp Feature Selection (BCOA)
    3. Dirichlet Non-IID Data Partitioning (3 clients, alpha=0.5)
    4. Trimmed Mean Server Aggregation (beta=0.1)
    5. Global Model & Metadata Checkpointing
    """
    print("=" * 70)
    print("      FedXAI-IDS: Intelligent Cybersecurity Framework Simulation      ")
    print("=" * 70)

    # 1. Dataset Preprocessing Check
    if not os.path.exists("data/ciciot2023_curated_subset.csv"):
        print("\n[STEP 1] Raw dataset curated subset missing.")
    else:
        print("\n[STEP 1] Curated subset found at data/ciciot2023_curated_subset.csv.")

    # 2. BCOA Feature Selection Check
    if not os.path.exists("data/ciciot2023_bcoa_selected.csv"):
        print(
            "\n[STEP 2] Running Binary Chimp Optimization (BCOA) feature selection..."
        )
        run_bcoa_feature_selection()
    else:
        print(
            "\n[STEP 2] BCOA filtered dataset found at data/ciciot2023_bcoa_selected.csv."
        )

    # 3. Non-IID Dirichlet Data Loaders
    print("\n[STEP 3] Partitioning dataset into Non-IID client DataLoaders...")
    client_dataloaders, global_priors, feature_names = load_and_partition_data(
        csv_path="data/ciciot2023_bcoa_selected.csv",
        num_clients=num_clients,
        alpha=alpha,
    )

    feature_dim = client_dataloaders[0]["feature_dim"]
    num_classes = client_dataloaders[0]["num_classes"]

    # 4. Initialize Global PyTorch Model & Flower Strategy
    print("\n[STEP 4] Setting up Trimmed Mean Aggregation Strategy (beta=0.1)...")
    init_model = IDS_MLP(input_dim=feature_dim, num_classes=num_classes)
    init_weights = [val.cpu().numpy() for _, val in init_model.state_dict().items()]
    init_parameters = fl.common.ndarrays_to_parameters(init_weights)

    strategy = TrimmedMeanStrategy(
        beta=0.1,
        initial_parameters=init_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    client_fn = create_client_fn(client_dataloaders)

    # 5. Start Flower Federated Learning Simulation
    print(
        f"\n[STEP 5] Launching Flower FL Simulation ({num_rounds} Rounds across {num_clients} Clients)..."
    )

    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    print("\n[SUCCESS] FL Simulation Completed!")

    # 6. Extract & Save Global Trained Model Weights
    print(f"\n[STEP 6] Saving global model checkpoint to {GLOBAL_MODEL_PATH}...")
    torch.save(init_model.state_dict(), GLOBAL_MODEL_PATH)

    # Save metadata for Streamlit UI
    metadata = {
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "feature_names": feature_names,
        "global_priors": global_priors.tolist(),
        "num_rounds": num_rounds,
        "num_clients": num_clients,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved system metadata to {METADATA_PATH}")
    return history


if __name__ == "__main__":
    run_fl_simulation(num_rounds=3, num_clients=3, alpha=0.5)
