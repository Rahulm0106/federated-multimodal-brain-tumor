import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import torch
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_ct_loader(data_dir, batch_size=32, shuffle=True):
    ct_dir = os.path.join(data_dir, 'CT')
    dataset = datasets.ImageFolder(ct_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_mri_loader(data_dir, batch_size=32, shuffle=True):
    mri_dir = os.path.join(data_dir, 'MRI')
    dataset = datasets.ImageFolder(mri_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class PairedMultimodalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.ct_dir = os.path.join(data_dir, 'CT')
        self.mri_dir = os.path.join(data_dir, 'MRI')
        self.classes = sorted(os.listdir(self.ct_dir))  # Assume ['glioma', ...]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        for cls in self.classes:
            ct_files = sorted([f for f in os.listdir(os.path.join(self.ct_dir, cls)) if f.endswith(('.jpg', '.png'))])
            mri_files = sorted([f for f in os.listdir(os.path.join(self.mri_dir, cls)) if f.endswith(('.jpg', '.png'))])
            min_len = min(len(ct_files), len(mri_files))
            for i in range(min_len):
                ct_path = os.path.join(self.ct_dir, cls, ct_files[i])
                mri_path = os.path.join(self.mri_dir, cls, mri_files[i])
                pairs.append((ct_path, mri_path, self.class_to_idx[cls]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mri_path, label = self.pairs[idx]
        ct_img = Image.open(ct_path).convert('RGB')
        mri_img = Image.open(mri_path).convert('RGB')
        if self.transform:
            ct_img = self.transform(ct_img)
            mri_img = self.transform(mri_img)
        return ct_img, mri_img, label

def get_paired_loader(data_dir, batch_size=32, shuffle=True):
    dataset = PairedMultimodalDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Toy test
if __name__ == "__main__":
    data_dir = '../../data/raw/Dataset'  # Relative to src/data
    loader = get_paired_loader(data_dir)
    for ct, mri, lbl in loader:
        print(ct.shape, mri.shape, lbl.shape)
        break