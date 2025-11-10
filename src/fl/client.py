# src/fl/client.py
import flwr as fl, torch, torch.nn as nn, numpy as np, yaml
from torch.optim import Adam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from src.models.cnn_ct import CNN_CT
from src.models.cnn_mri import CNN_MRI
from src.data.preprocess import create_dataloaders, create_paired_dataloaders
from src.utils.logger import logger   # you can define a simple logger if missing

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, cfg_path='configs/default.yaml'):
        with open(cfg_path) as f: cfg = yaml.safe_load(f)
        self.cid = cid
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_ct  = CNN_CT(cfg['num_classes']).to(self.device)
        self.cnn_mri = CNN_MRI(cfg['num_classes']).to(self.device)
        self.opt_ct  = Adam(self.cnn_ct.parameters(),  lr=cfg['lr'])
        self.opt_mri = Adam(self.cnn_mri.parameters(), lr=cfg['lr'])
        self.crit = nn.CrossEntropyLoss()

        self.train_ct,  _ = create_dataloaders('ct')
        self.train_mri, _ = create_dataloaders('mri')
        _, self.val_paired = create_paired_dataloaders()   # use val as test set

        self.epochs = cfg['epochs']
        self.classifier = None

    # ---------- parameter handling ----------
    def get_parameters(self, config):
        ct  = [v.cpu().numpy() for v in self.cnn_ct.state_dict().values()]
        mri = [v.cpu().numpy() for v in self.cnn_mri.state_dict().values()]
        return ct + mri

    def set_parameters(self, parameters):
        n_ct = len(self.cnn_ct.state_dict())
        ct_params  = [torch.tensor(p) for p in parameters[:n_ct]]
        mri_params = [torch.tensor(p) for p in parameters[n_ct:]]
        ct_state  = dict(zip(self.cnn_ct.state_dict().keys(),  ct_params))
        mri_state = dict(zip(self.cnn_mri.state_dict().keys(), mri_params))
        self.cnn_ct.load_state_dict(ct_state)
        self.cnn_mri.load_state_dict(mri_state)

    # ---------- local training ----------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.cnn_ct.train(); self.cnn_mri.train()
        for epoch in range(self.epochs):
            # CT
            for x, y in self.train_ct:
                x, y = x.to(self.device), y.to(self.device)
                self.opt_ct.zero_grad()
                loss = self.crit(self.cnn_ct(x), y)
                loss.backward()
                self.opt_ct.step()
            # MRI
            for x, y in self.train_mri:
                x, y = x.to(self.device), y.to(self.device)
                self.opt_mri.zero_grad()
                loss = self.crit(self.cnn_mri(x), y)
                loss.backward()
                self.opt_mri.step()
        n_samples = len(self.train_ct.dataset) + len(self.train_mri.dataset)
        return self.get_parameters({}), n_samples, {}

    # ---------- feature extraction & SVC ----------
    def _extract_fused(self, loader):
        self.cnn_ct.eval(); self.cnn_mri.eval()
        feats, labs = [], []
        with torch.no_grad():
            for ct, mri, y in loader:
                ct, mri = ct.to(self.device), mri.to(self.device)
                f_ct  = self.cnn_ct.extract_features(ct)
                f_mri = self.cnn_mri.extract_features(mri)
                fused = torch.cat([f_ct, f_mri], dim=1).cpu().numpy()
                feats.append(fused); labs.append(y.numpy())
        return np.vstack(feats), np.hstack(labs)

    def fit_classifier(self):
        X, y = self._extract_fused(self.val_paired)
        self.classifier = SVC(kernel='linear', probability=True)
        self.classifier.fit(X, y)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.fit_classifier()
        X, y = self._extract_fused(self.val_paired)
        pred = self.classifier.predict(X)
        acc = accuracy_score(y, pred)
        f1  = f1_score(y, pred, average='macro')
        logger.info(f"Client {self.cid} – Acc: {acc:.4f}  F1: {f1:.4f}")
        return 0.0, len(y), {"accuracy": acc, "f1": f1}