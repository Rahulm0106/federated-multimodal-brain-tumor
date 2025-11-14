# src/fl/client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.models.local_heads import RandomForestHead
from src.data.preprocess import create_dataloaders


class FLClient:
    def __init__(self, client_id: int, modality: str, config):
        self.client_id = client_id
        self.modality = modality
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_loader, self.val_loader = create_dataloaders(modality)

        # Model
        from src.models import get_model
        self.model = get_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Local non-gradient head
        rf_cfg = config["client"]
        self.local_head = RandomForestHead(
            n_estimators=rf_cfg["rf_n_estimators"],
            max_depth=rf_cfg["rf_max_depth"]
        )
        self.model.set_local_head(self.local_head)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.get_gradient_params(), parameters)
        state_dict = {k: v for k, v in params_dict}
        self.model.body.load_state_dict(state_dict, strict=True)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.body.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.AdamW(self.model.get_gradient_params(), lr=1e-3)
        epochs = config.get("local_epochs", 3)

        # Train backbone with cross-entropy
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                features = self.model(images)
                loss = self.criterion(features, labels)
                loss.backward()
                optimizer.step()

        # Train local RF on frozen features
        self._train_local_rf()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def _train_local_rf(self):
        self.model.eval()
        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(self.device)
                feats = self.model.get_features(images)
                features_list.append(feats.cpu())
                labels_list.append(labels)
        features = torch.cat(features_list)
        labels = torch.cat(labels_list)
        self.local_head.fit(features, labels)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                feats = self.model.get_features(images)
                features_list.append(feats.cpu())
                labels_list.append(labels)
        features = torch.cat(features_list)
        labels = torch.cat(labels_list)

        preds = self.local_head.predict_class(features)
        acc = accuracy_score(labels.numpy(), preds.numpy())

        return 0.0, len(labels), {"accuracy": float(acc)}