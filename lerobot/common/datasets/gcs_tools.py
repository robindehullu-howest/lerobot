import logging
from google.cloud import storage

from lerobot.common.constants import HF_LEROBOT_HOME
import argparse

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Pull or push dataset from/to GCS bucket.")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the GCS bucket.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID to pull or push.")
    parser.add_argument("--action", type=str, choices=["pull", "push"], required=True, help="Action to perform: 'pull' or 'push'.")
    return parser.parse_args()

def pull_dataset_from_gcs(bucket_name, repo_id) -> None:
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
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded gs://{bucket_name}/{blob.name} to {local_path}")

def push_dataset_to_gcs(bucket_name, repo_id) -> None:
    """
    Uploads the entire dataset directory from the local cache to the specified GCS bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    root = HF_LEROBOT_HOME / repo_id

    for local_path in root.rglob("*"):
        if local_path.is_dir():
            continue

        relative_path = local_path.relative_to(root)
        blob = bucket.blob(f"{repo_id}/{relative_path}")
        blob.upload_from_filename(local_path)
        logging.info(f"Uploaded {local_path} to gs://{bucket_name}/{repo_id}/{relative_path}")

if __name__ == "__main__":
    args = parse_args()
    bucket_name = args.bucket_name
    repo_id = args.repo_id
    action = args.action

    if action == "pull":
        pull_dataset_from_gcs(bucket_name, repo_id)
    elif action == "push":
        push_dataset_to_gcs(bucket_name, repo_id)