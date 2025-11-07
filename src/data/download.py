import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()  # Uses ~/.kaggle/kaggle.json
    dataset_slug = 'murtozalikhon/brain-tumor-multimodal-image-ct-and-mri'
    download_path = 'data/raw'
    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading dataset to {download_path}...")
    api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
    print("Download complete! Check data/raw/ for files.")

if __name__ == "__main__":
    download_dataset()
