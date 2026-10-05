import torch
import numpy as np
import pandas as pd
import shap
from src.model import IDS_MLP

CLASS_NAMES = [
    "Benign",
    "DDoS",
    "DoS",
    "Recon",
    "Spoofing",
    "BruteForce",
    "Web_Malware",
]


class IDSExplainer:
    """
    Explainable AI (XAI) Feature Attribution Module for FedXAI-IDS using SHAP.

    Provides decision transparency for individual intrusion detection predictions
    by calculating local feature attribution scores via Gradient/Deep SHAP.
    """

    def __init__(self, model: IDS_MLP, feature_names: list, class_names: list = None):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.class_names = class_names if class_names else CLASS_NAMES

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32)
        return data

    def compute_shap_values(self, background_data: np.ndarray, sample_data: np.ndarray):
        """
        Computes SHAP feature attribution values for sample_data using background_data baseline.
        """
        bg_tensor = self._to_tensor(background_data)
        sample_tensor = self._to_tensor(sample_data)

        try:
            # DeepExplainer for PyTorch neural networks
            explainer = shap.DeepExplainer(self.model, bg_tensor)
            shap_values = explainer.shap_values(sample_tensor)
        except Exception:
            # Fallback to GradientExplainer or KernelExplainer
            explainer = shap.GradientExplainer(self.model, bg_tensor)
            shap_values = explainer.shap_values(sample_tensor)

        return shap_values

    def explain_sample(
        self, sample_vector: np.ndarray, background_data: np.ndarray, top_k: int = 5
    ) -> dict:
        """
        Explains a single network flow sample vector and returns top-k feature attributions.
        """
        if sample_vector.ndim == 1:
            sample_vector = sample_vector.reshape(1, -1)

        # Model forward pass for prediction & class probabilities
        sample_tensor = torch.tensor(sample_vector, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(sample_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_class_idx = int(np.argmax(probs))

        pred_label = self.class_names[pred_class_idx]
        confidence = float(probs[pred_class_idx])

        # Compute SHAP attributions
        shap_vals = self.compute_shap_values(background_data, sample_vector)

        # Unpack SHAP array for predicted class
        if isinstance(shap_vals, list):
            class_shap = shap_vals[pred_class_idx][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            class_shap = shap_vals[0, :, pred_class_idx]
        else:
            class_shap = shap_vals[0]

        # Rank top features by absolute attribution magnitude
        feature_importance = [
            {
                "feature": self.feature_names[i],
                "val": float(sample_vector[0, i]),
                "shap_val": float(class_shap[i]),
                "abs_shap": abs(float(class_shap[i])),
            }
            for i in range(len(self.feature_names))
        ]

        # Sort descending by absolute SHAP score
        feature_importance.sort(key=lambda x: x["abs_shap"], reverse=True)
        top_features = feature_importance[:top_k]

        return {
            "pred_class_idx": pred_class_idx,
            "pred_label": pred_label,
            "confidence": confidence,
            "all_probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
            "top_features": top_features,
        }


if __name__ == "__main__":
    # Quick sanity check test
    model = IDS_MLP(input_dim=15, num_classes=7)
    bg = np.random.randn(20, 15).astype(np.float32)
    sample = np.random.randn(1, 15).astype(np.float32)
    f_names = [f"Feature_{i}" for i in range(15)]

    explainer = IDSExplainer(model, f_names)
    explanation = explainer.explain_sample(sample, bg, top_k=5)
    print(
        f"[XAI CHECK] Pred Label: {explanation['pred_label']} (Conf: {explanation['confidence']:.2%})"
    )
    print("Top Features:", explanation["top_features"])
