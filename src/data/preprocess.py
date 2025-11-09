# src/data/preprocess.py
"""
Unified preprocessing for CT and MRI brain tumor datasets.
"""

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
BATCH_SIZE = 32
LABELS = ["Healthy", "Tumor"]


def get_root_directory(modality: str) -> str:
    """Get root directory based on modality."""
    if modality == "ct":
        return "data/raw/Dataset/Brain Tumor CT scan Images"
    elif modality == "mri":
        return "data/raw/Dataset/Brain Tumor MRI images"
    raise ValueError("Invalid modality. Choose 'ct' or 'mri'.")


def collect_filepaths_and_labels(root_dir: str) -> pd.DataFrame:
    """Collect filepaths and labels."""
    filepaths = []
    labels = []
    for dirpath, _, filenames in os.walk(root_dir):
        label = os.path.basename(dirpath)
        if label in LABELS:
            for filename in filenames:
                filepaths.append(os.path.join(dirpath, filename))
                labels.append(label)
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    print(df.head())
    print(f"\nFound {len(df)} images.")
    return df


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get transforms (shared for both modalities)."""
    common = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ] + common)
    return transforms.Compose(common)


class BrainScanDataset(Dataset):
    """Unified dataset class for brain scans."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform
        self.label_map = {"Healthy": 0, "Tumor": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple:
        img_path = self.df.iloc[index]["filepath"]
        label_str = self.df.iloc[index]["label"]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        return image, label


def create_dataloaders(modality: str):
    """Create train/val dataloaders for the given modality."""
    df = collect_filepaths_and_labels(get_root_directory(modality))
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_ds = BrainScanDataset(df_train, get_transforms(train=True))
    val_ds = BrainScanDataset(df_val, get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Created {len(train_ds)} train and {len(val_ds)} val samples for {modality.upper()}.")
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mri"], required=True)
    args = parser.parse_args()
    create_dataloaders(args.modality)