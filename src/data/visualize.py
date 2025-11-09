# src/data/visualize.py
"""
Unified visualization for brain scan datasets.
"""

import matplotlib.pyplot as plt
import torchvision
from preprocess import create_dataloaders  # Import from preprocess.py


def visualize_batch(modality: str):
    """Visualize a sample batch."""
    train_loader, _ = create_dataloaders(modality)
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        print("DataLoader is empty.")
        return

    label_map = {0: "Healthy", 1: "Tumor"}
    batch_labels = [label_map[l.item()] for l in labels]

    img_grid = torchvision.utils.make_grid(images, nrow=8)
    np_img = img_grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(15, 10))
    plt.imshow(np_img)
    plt.title(f"Sample {modality.upper()} Batch\nLabels: {batch_labels[:32]}", fontsize=10)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mri"], required=True)
    args = parser.parse_args()
    visualize_batch(args.modality)