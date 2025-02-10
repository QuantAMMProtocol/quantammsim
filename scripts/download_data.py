import os
import hashlib
import gdown
import zipfile
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_FILES = {
    "data.zip": {
        "id": "13fGnXMB5Qep_B8hWH-ucB-rH2S63VpzY",
        "size": 1679592725,
        "sha256": "1c3401b98377d38ccd313309409479145fb1ffef1853fa598f30755d1b54ade0",
    },
}

DATA_DIR = Path("../quantammsim/data/")


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(file_id, dest_path, expected_hash):
    # Check if complete file exists and is valid
    if dest_path.exists():
        if calculate_sha256(dest_path) == expected_hash:
            print(f"File {dest_path.name} already exists and is valid.")
            return True
        dest_path.unlink()

    # Download using gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(dest_path), quiet=False)

        # Verify download
        if calculate_sha256(dest_path) == expected_hash:
            return True
        else:
            dest_path.unlink()
            return False
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_zip(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            total_size = sum(file.file_size for file in zip_ref.filelist)
            extracted_size = 0

            for file in zip_ref.filelist:
                zip_ref.extract(file, extract_path)
                extracted_size += file.file_size
                progress = (extracted_size / total_size) * 100
                print(f"\rExtracting: {progress:.1f}%", end="")

        print("\nExtraction completed successfully")
        zip_path.unlink()
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def main():
    DATA_DIR.mkdir(exist_ok=True)

    for filename, info in DATA_FILES.items():
        print(f"\nDownloading {filename}...")
        dest_path = DATA_DIR / filename

        success = download_file(info["id"], dest_path, info["sha256"])
        if not success:
            print(f"Failed to download {filename}")
            return

        print(f"Extracting {filename}...")
        if not extract_zip(dest_path, DATA_DIR):
            print(f"Failed to extract {filename}")
            return

    print("\nAll files downloaded and extracted successfully!")


if __name__ == "__main__":
    main()
