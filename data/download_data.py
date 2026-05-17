"""Download and extract the MovieLens 100K dataset into data/raw/."""
import os
import zipfile
import requests

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-100k.zip")

    if os.path.exists(os.path.join(RAW_DIR, "ml-100k", "u.data")):
        print("Dataset already downloaded.")
        return

    print("Downloading MovieLens 100K...")
    response = requests.get(URL, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    os.remove(zip_path)
    print(f"Dataset ready at {RAW_DIR}/ml-100k/")


if __name__ == "__main__":
    download()
