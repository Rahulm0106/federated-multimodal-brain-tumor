# 🧠 Data Management — Federated Multimodal Brain Tumor Classification

This folder contains raw and processed datasets for the Federated Multimodal Brain Tumor Classification project.

---

## 📁 Folder Structure

```
data/
├── raw/
│   └── Dataset/
│       ├── Brain Tumor CT scan Images/
│       │   ├── Healthy/
│       │   └── Tumor/
│       └── Brain Tumor MRI Images/
│           ├── Healthy/
│           └── Tumor/
└── processed/
```

- **`raw/`** — Stores downloaded datasets from Kaggle (CT and MRI images)
- **`processed/`** — Stores preprocessed data (e.g., tensors or augmented files)
  - _Initially empty — populated via preprocessing scripts_

---

## 📥 Downloading Data

Download the multimodal brain tumor dataset (CT + MRI images) from Kaggle:

```bash
python src/data/download.py
```

### ⚙️ Requirements

- Kaggle API key stored at `~/.kaggle/kaggle.json`
- Internet access to fetch dataset

Downloaded files unzip to:

```
data/raw/Dataset/
```

### Dataset Summary

- **~4,618 CT images**
- **~5,000 MRI images**
- **Binary labels:** Healthy / Tumor

---

## ⚙️ Preprocessing Data

Unified preprocessing for CT and MRI datasets includes:

- Collecting image paths
- Applying transforms (resize 224×224, grayscale, augmentations)
- Splitting into 80/20 train/validation
- Creating PyTorch DataLoaders

### Run Preprocessing

**For CT images:**

```bash
python src/data/preprocess.py --modality ct
```

**For MRI images:**

```bash
python src/data/preprocess.py --modality mri
```

### Output

- In-memory DataLoaders
- Printed dataset statistics and batch shapes

---

## 🖼️ Visualizing Data

Display a sample batch grid with labels.

**For CT images:**

```bash
python src/data/visualize.py --modality ct
```

**For MRI images:**

```bash
python src/data/visualize.py --modality mri
```

> **Note:** Requires Matplotlib. Shows 32 images in a single figure with corresponding labels.

---

## 🧩 Notes

- Run `download.py` before running preprocessing or visualization scripts
- You can modify scripts under `src/data/` to adjust parameters like batch size or augmentation
- Processed DataLoaders are utilized in federated learning experiments via:

---

## 📌 Quick Reference

| Task | Command |
|------|---------|
| Download datasets | `python src/data/download.py` |
| Preprocess CT data | `python src/data/preprocess.py --modality ct` |
| Preprocess MRI data | `python src/data/preprocess.py --modality mri` |
| Visualize CT data | `python src/data/visualize.py --modality ct` |
| Visualize MRI data | `python src/data/visualize.py --modality mri` |