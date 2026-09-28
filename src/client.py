import collections
import numpy as np
import torch
import torch.optim as optim
import flwr as fl
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.model import IDS_MLP, LogitAdjustedLoss

class IDSClient(fl.client.NumPyClient):
    """
    Flower Client Wrapper for FedXAI-IDS Edge Nodes.
    
    Manages local PyTorch model training with Logit Adjustment Loss over Non-IID client 
    partitions, extracts weight parameters into NumPy arrays for network transmission, 
    and computes local intrusion classification evaluation metrics (Loss, Accuracy, Precision, Recall, F1).
    """
    def __init__(self, 
                 client_id: int, 
                 dataloaders: dict, 
                 device: torch.device = None):
        self.client_id = client_id
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.test_loader = dataloaders["test"]
        self.feature_dim = dataloaders["feature_dim"]
        self.num_classes = dataloaders["num_classes"]
        self.class_priors = dataloaders["class_priors"]
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate PyTorch Model
        self.model = IDS_MLP(input_dim=self.feature_dim, num_classes=self.num_classes).to(self.device)
        
        # Instantiate Logit Adjustment Loss Criterion & Adam Optimizer
        self.criterion = LogitAdjustedLoss(cls_priors=self.class_priors, tau=1.0).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

    def get_parameters(self, config=None):
        """Extract PyTorch state_dict parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load central server parameters (list of NumPy arrays) into local PyTorch state_dict."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Local PyTorch Training Loop for FL round."""
        self.set_parameters(parameters)
        self.model.train()
        
        local_epochs = config.get("local_epochs", 3) if config else 3
        running_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
                
        epoch_loss = running_loss / (total * local_epochs)
        epoch_acc = correct / (total * local_epochs)
        
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(epoch_loss),
            "train_acc": float(epoch_acc)
        }
        
        return self.get_parameters(), total, metrics

    def evaluate(self, parameters, config):
        """Local Evaluation Loop on Client Test Set."""
        self.set_parameters(parameters)
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        num_samples = len(all_targets)
        eval_loss = running_loss / num_samples
        acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            "client_id": self.client_id,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }
        
        return float(eval_loss), num_samples, metrics

def create_client_fn(client_dataloaders: dict):
    """Factory function creating Flower NumPyClient instances per client ID."""
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        return IDSClient(client_id=client_id, dataloaders=client_dataloaders[client_id]).to_client()
    return client_fn
