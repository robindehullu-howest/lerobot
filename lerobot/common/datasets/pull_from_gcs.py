import logging
from google.cloud import storage

from lerobot.common.constants import HF_LEROBOT_HOME
import argparse

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Pull dataset from GCS bucket.")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the GCS bucket.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID to pull from the bucket.")
    return parser.parse_args()

def pull_from_gcs(bucket_name, repo_id) -> None:
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

if __name__ == "__main__":
    args = parse_args()
    bucket_name = args.bucket_name
    repo_id = args.repo_id

    pull_from_gcs(bucket_name, repo_id)