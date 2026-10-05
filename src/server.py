from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg


class TrimmedMeanStrategy(FedAvg):
    """
    Byzantine-Robust Coordinate-Wise Trimmed Mean Strategy for FedXAI-IDS.

    Protects central server model parameter aggregation against model poisoning
    and Byzantine updates by sorting coordinate weights across client models
    and trimming the top and bottom beta fraction (default 10%) before averaging.

    Prevents adversarial weight corruptions without requiring clean server validation datasets.
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[
            Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]
        ],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregates local model updates using coordinate-wise Trimmed Mean."""
        if not results:
            return None, {}
        if self.accept_failures and failures:
            print(
                f"[SERVER WARNING] Round {server_round}: {len(failures)} client failures detected."
            )

        # Convert parameters from client FitRes to list of NumPy ndarrays per client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        num_clients = len(weights_results)
        if num_clients == 0:
            return None, {}

        # Number of clients to trim from top and bottom
        k = int(np.floor(self.beta * num_clients))

        # Unpack parameters: list of layer weight arrays for each client
        num_layers = len(weights_results[0][0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            # Stack layer weights across all clients -> shape: (num_clients, *layer_shape)
            layer_updates = np.stack(
                [weights_results[c][0][layer_idx] for c in range(num_clients)], axis=0
            )

            if k > 0 and num_clients > 2 * k:
                # Sort along client dimension (axis=0)
                sorted_updates = np.sort(layer_updates, axis=0)
                # Trim top and bottom k updates per coordinate
                trimmed_updates = sorted_updates[k : num_clients - k]
                # Compute coordinate-wise mean of remaining updates
                layer_mean = np.mean(trimmed_updates, axis=0)
            else:
                # Fallback to standard weighted average if too few clients for trimming
                total_samples = sum(num_examples for _, num_examples in weights_results)
                layer_mean = sum(
                    layer_updates[c] * (num_examples / total_samples)
                    for c, (_, num_examples) in enumerate(weights_results)
                )

            aggregated_weights.append(layer_mean)

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate training metrics
        metrics_aggregated = {}
        if results:
            train_accs = [
                fit_res.metrics.get("train_acc", 0.0) for _, fit_res in results
            ]
            metrics_aggregated["mean_train_acc"] = float(np.mean(train_accs))

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[
            Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]
        ],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregates evaluation metrics across client test sets."""
        if not results:
            return None, {}

        # Compute sample-weighted average loss
        total_samples = sum(eval_res.num_examples for _, eval_res in results)
        weighted_loss = (
            sum(eval_res.loss * eval_res.num_examples for _, eval_res in results)
            / total_samples
        )

        # Aggregate metrics
        accuracies = [
            eval_res.metrics.get("accuracy", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ]
        precisions = [
            eval_res.metrics.get("precision", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ]
        recalls = [
            eval_res.metrics.get("recall", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ]
        f1s = [
            eval_res.metrics.get("f1", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ]

        metrics_aggregated = {
            "accuracy": float(sum(accuracies) / total_samples),
            "precision": float(sum(precisions) / total_samples),
            "recall": float(sum(recalls) / total_samples),
            "f1": float(sum(f1s) / total_samples),
        }

        print(
            f"[ROUND {server_round} GLOBAL METRICS] Loss: {weighted_loss:.4f} | Accuracy: {metrics_aggregated['accuracy']:.4f} | F1-Score: {metrics_aggregated['f1']:.4f}"
        )

        return float(weighted_loss), metrics_aggregated


if __name__ == "__main__":
    # Quick sanity check test
    strategy = TrimmedMeanStrategy(beta=0.1)
    print("[SERVER MODULE CHECK] TrimmedMeanStrategy initialized successfully.")
