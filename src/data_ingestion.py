import requests
import logging
from pathlib import Path
from tqdm import tqdm

# ------------------
# Logging setup
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------
# Dataset config
# ------------------
DATASETS = [
    {
        "name": "spanish",
        "url": "https://zenodo.org/record/3904280/files/FETAL_PLANES_ZENODO.zip?download=1",
        "filename": "FETAL_PLANES_ZENODO.zip",
    },
    {
        "name": "african",
        "url": "https://zenodo.org/record/7540448/files/Zenodo_dataset.tar.xz?download=1",
        "filename": "Zenodo_dataset.tar.xz",
    },
]

BASE_DIR = Path.cwd() / "data" / "raw"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_file(url: str, output_path: Path):
    logger.info(f"Starting download: {output_path.name}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Download completed: {output_path}")


def main():
    for ds in DATASETS:
        output_dir = BASE_DIR / ds["name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / ds["filename"]

        if output_path.exists():
            logger.info(f"{output_path.name} already exists. Skipping download.")
            continue

        download_file(ds["url"], output_path)


if __name__ == "__main__":
    main()
