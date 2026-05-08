import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = "Cornell-University/arxiv"
    download_path = "./data"
    
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading {dataset_name} to {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    
    print("Download and extraction complete. The metadata JSON is in the 'data' folder.")

if __name__ == "__main__":
    main()
