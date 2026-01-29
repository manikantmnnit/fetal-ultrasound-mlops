import zipfile
import tarfile
import logging
from pathlib import Path


# ------------------
# Logging
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path.cwd() / "data" / "raw"

DATASETS = [
    {
        "name": "spanish",
        "file": BASE_DIR / "spanish" / "FETAL_PLANES_ZENODO.zip",
        "type": "zip",
    },
    {
        "name": "african",
        "file": BASE_DIR / "african" / "Zenodo_dataset.tar.xz",
        "type": "tar.xz",
    },
]


def extract_archive(dataset: dict, output_dir: Path):
    archive = dataset["file"]

    if not archive.exists():
        raise FileNotFoundError(f"{archive} not found")

    logger.info(f"Extracting {archive.name}")

    if dataset["type"] == "zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(output_dir)
           

    elif dataset["type"] == "tar.xz":
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(output_dir)

    else:
        raise ValueError(f"Unsupported archive type: {dataset['type']}")


def main():
    for dataset in DATASETS:
        output_dir = BASE_DIR .parent/"extracted"/dataset["name"] 

        if output_dir.exists():
            logger.info(f"{dataset['name']} already extracted. Skipping.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(dataset, output_dir)

        logger.info(f"Extraction completed: {output_dir}")


if __name__ == "__main__":
    main()
