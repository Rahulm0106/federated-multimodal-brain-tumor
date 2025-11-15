import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import flwr as fl
import numpy as np
from typing import List

from src.models.local_heads import RandomForestHead
from src.data.preprocess import create_dataloaders
from src.models import get_model


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, modality: str, config):
        self.client_id = client_id
        self.modality = modality.upper()
        self.config = config
        # Enable MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_loader, self.val_loader = create_dataloaders(modality)

        self.model = get_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        rf_cfg = config["client"]
        self.local_head = RandomForestHead(
            n_estimators=rf_cfg["rf_n_estimators"],
            max_depth=rf_cfg["rf_max_depth"]
        )
        self.model.set_local_head(self.local_head)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.body.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.body.state_dict()
        new_state_dict = {}
        for (name, _), ndarray in zip(state_dict.items(), parameters):
            new_state_dict[name] = torch.tensor(ndarray).to(self.device)
        self.model.body.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.AdamW(self.model.get_gradient_params(), lr=1e-3)
        epochs = config.get("local_epochs", 3)

        print(f"\nClient {self.client_id} ({self.modality}) | Training {epochs} epochs...")
        for epoch in tqdm(range(epochs), desc=f"C{self.client_id} Train", leave=False):
            for images, labels in self.train_loader:
                images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                features = self.model(images)
                loss = self.criterion(features, labels)
                loss.backward()
                optimizer.step()

        print(f"Client {self.client_id} | Fitting local RF head...")
        self._train_local_rf()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def _train_local_rf(self):
        self.model.eval()
        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.train_loader, desc=f"C{self.client_id} RF", leave=False):
                images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
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
                images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
                images = images.to(self.device)
                feats = self.model.get_features(images)
                features_list.append(feats.cpu())
                labels_list.append(labels)

        features = torch.cat(features_list)
        labels = torch.cat(labels_list)
        preds = self.local_head.predict_class(features)
        acc = accuracy_score(labels.numpy(), preds)

        print(f"Client {self.client_id} ({self.modality}) | Val Acc: {acc:.4f}")
        return 0.0, len(labels), {"accuracy": float(acc)}