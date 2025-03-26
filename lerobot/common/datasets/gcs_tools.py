import logging
from google.cloud import storage

from lerobot.common.constants import HF_LEROBOT_HOME
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)

MODEL_OUTPUT_DIR = Path("outputs/train")

def parse_args():
    parser = argparse.ArgumentParser(description="Pull or push dataset from/to GCS bucket.")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the GCS bucket.")
    parser.add_argument("--action", type=str, choices=["pull", "push"], required=True, help="Action to perform: 'pull' or 'push'.")
    parser.add_argument("--content_type", type=str, choices=["dataset", "model"], required=True, help="Type of content to pull or push: 'dataset' or 'model'.")
    parser.add_argument("--identifier", type=str, required=True, help="Repository ID to pull or push.")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()

def pull_dataset_from_gcs(bucket_name: str, repo_id: str, force_overwrite: bool = False) -> None:
    """
    Downloads the entire dataset directory from the specified GCS bucket to the local cache.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    root = HF_LEROBOT_HOME / repo_id

    prefix = f"{repo_id}/"
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        relative_path = blob.name[len(prefix):]
        local_path = root / relative_path

        if not force_overwrite and local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded {blob.name} to {local_path}")

def push_dataset_to_gcs(bucket_name: str, repo_id: str, force_overwrite: bool = False) -> None:
    """
    Uploads the entire dataset directory from the local cache to the specified GCS bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    root = HF_LEROBOT_HOME / repo_id

    for local_path in root.rglob("*"):
        if local_path.is_dir():
            continue

        relative_path = local_path.relative_to(root).as_posix()
        blob_name = f"{repo_id}/{relative_path}"
        blob = bucket.blob(blob_name)

        parent_dir_name = Path(relative_path).parent.name
        if not force_overwrite and blob.exists() and parent_dir_name != "meta":
            continue
        
        blob.upload_from_filename(local_path)
        logging.info(f"Uploaded {blob_name} to {bucket_name}")

def pull_model_from_gcs(bucket_name: str, model_name: str, force_overwrite: bool = False) -> None:
    """
    Downloads the model from the specified GCS bucket to the local cache.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    prefix = f"{model_name}/"
    blobs = bucket.list_blobs(prefix=prefix)
    
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        
        local_path = MODEL_OUTPUT_DIR / blob.name

        if not force_overwrite and local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded gs://{bucket_name}/{blob.name} to {local_path}")

def push_model_to_gcs(bucket_name: str, model_name: str, force_overwrite: bool = False) -> None:
    """
    Uploads the model from the local cache to the specified GCS bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    root = MODEL_OUTPUT_DIR / model_name

    for local_path in root.rglob("*"):
        if local_path.is_dir():
            continue

        relative_path = local_path.relative_to(root).as_posix()
        blob = bucket.blob(f"{model_name}/{relative_path}")

        if not force_overwrite and blob.exists():
            continue

        blob.upload_from_filename(local_path)
        logging.info(f"Uploaded {local_path} to gs://{bucket_name}/{model_name}/{relative_path}")

if __name__ == "__main__":
    args = parse_args()
    bucket_name = args.bucket_name
    identifier = args.identifier
    action = args.action
    force_overwrite = args.force_overwrite

    if args.content_type == "dataset":
        if action == "pull":
            pull_dataset_from_gcs(bucket_name, identifier, force_overwrite)
        elif action == "push":
            push_dataset_to_gcs(bucket_name, identifier, force_overwrite)
    elif args.content_type == "model":
        if action == "pull":
            pull_model_from_gcs(bucket_name, identifier, force_overwrite)
        elif action == "push":
            push_model_to_gcs(bucket_name, identifier, force_overwrite)