import os
import logging
import argparse
from pathlib import Path
from google.cloud import storage
from typing import List
from concurrent.futures import ThreadPoolExecutor, wait

def parse_args():
    parser = argparse.ArgumentParser(description="Pull or push dataset from/to GCS bucket.")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the GCS bucket.")
    parser.add_argument("--action", type=str, choices=["pull", "push"], required=True, help="Action to perform: 'pull' or 'push'.")
    parser.add_argument("--content_type", type=str, choices=["dataset", "model"], required=True, help="Type of content to pull or push: 'dataset' or 'model'.")
    parser.add_argument("--identifiers", type=str, required=True, help="Repository IDs delimited by a comma to pull or push.")
    parser.add_argument("--base_dir", type=str, help="Base directory for the dataset or model.")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def download_blob(blob, local_path):
    """Helper function to download a single blob."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    logging.info(f"Downloaded {blob.name} to {local_path}")

def upload_blob(local_path, blob):
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded {blob.name} to {bucket_name}")


def pull_datasets_from_gcs(bucket_name: str, base_dir: str, dataset_ids: List[str], force_overwrite: bool = False) -> None:
    """
    Downloads the entire dataset directory from the specified GCS bucket to the local cache.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)


    for dataset_id in dataset_ids:
        logging.info(f"Pulling dataset {dataset_id}.")
        blobs = bucket.list_blobs(prefix=dataset_id)

        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
            for blob in blobs:
                local_path = Path(base_dir, blob.name)
                parent_dir_name = local_path.parent.name

                if not force_overwrite and local_path.exists() and parent_dir_name != "meta":
                    continue

                futures.append(executor.submit(download_blob, blob, local_path))

            wait(futures)


def push_datasets_to_gcs(bucket_name: str, base_dir: str, dataset_ids: List[str], force_overwrite: bool = False) -> None:
    """
    Uploads the entire dataset directory from the local cache to the specified GCS bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for dataset_id in dataset_ids:
        dataset_dir = Path(base_dir, dataset_id)
        futures = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            for local_path in dataset_dir.rglob("*"):
                if local_path.is_dir():
                    continue

                blob_name = local_path.relative_to(base_dir).as_posix()
                blob = bucket.blob(blob_name)

                # Check if the blob already exists and if we should overwrite it
                # Always overwrite meta files
                if not force_overwrite and blob.exists() and local_path.parent.name != "meta":
                    continue

                # Skip stats.json files (auto-generated)
                if local_path.name == "stats.json":
                    continue

                futures.append(executor.submit(upload_blob, local_path, blob))
            wait(futures)


def pull_models_from_gcs(bucket_name: str, base_dir: str, model_ids: List[str], force_overwrite: bool = False) -> None:
    """
    Downloads the model from the specified GCS bucket to the local cache.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for model_id in model_ids:
        blobs = bucket.list_blobs(prefix=model_id)

        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for blob in blobs:
                if blob.name.endswith(('/', '.pt', '.pth')):
                    continue

                local_path = Path(base_dir, blob.name)

                if not force_overwrite and local_path.exists():
                    continue

                futures.append(executor.submit(download_blob, blob, local_path))

            wait(futures)


def push_models_to_gcs(bucket_name: str, base_dir: Path | str, model_ids: List[str], force_overwrite: bool = False) -> None:
    """
    Uploads the model from the local cache to the specified GCS bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for model_id in model_ids:
        model_dir = Path(base_dir, model_id)
        futures = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            for local_path in model_dir.rglob("*"):
                if local_path.is_dir() or local_path.name.endswith(('.pt', '.pth')):
                    continue

                blob_name = local_path.relative_to(base_dir).as_posix()
                blob = bucket.blob(blob_name)

                if not force_overwrite and blob.exists():
                    continue

                futures.append(executor.submit(upload_blob, local_path, blob))
            wait(futures)


if __name__ == "__main__":
    cache_dir = Path(os.getenv("HOME") or os.getenv("USERPROFILE"), ".cache", "huggingface", "lerobot")
    output_dir = Path("outputs/train")

    args = parse_args()
    bucket_name = args.bucket_name
    identifiers = args.identifiers.split(",")
    action = args.action
    force_overwrite = args.force_overwrite
    base_dir = args.base_dir

    if not base_dir:
        base_dir = cache_dir if args.content_type == "dataset" else output_dir

    if args.content_type == "dataset":
        if action == "pull":
            pull_datasets_from_gcs(bucket_name, base_dir, identifiers, force_overwrite)
        elif action == "push":
            push_datasets_to_gcs(bucket_name, base_dir, identifiers, force_overwrite)
    elif args.content_type == "model":
        if action == "pull":
            pull_models_from_gcs(bucket_name, base_dir, identifiers, force_overwrite)
        elif action == "push":
            push_models_to_gcs(bucket_name, base_dir, identifiers, force_overwrite)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)