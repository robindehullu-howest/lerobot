import logging
import argparse
import os
import json
import shutil
import jsonlines
from typing import List, Optional, Dict, Any

from lerobot.common.constants import HF_LEROBOT_HOME

META_DIR = "meta"
INFO_FILE = "info.json"
TASKS_FILE = "tasks.jsonl"
EPISODES_FILE = "episodes.jsonl"
EPISODES_STATS_FILE = "episodes_stats.jsonl"
DATA_DIR = "data"
VIDEO_DIR = "videos"

logging.basicConfig(level=logging.INFO)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Combine datasets from specified repositories.")
    parser.add_argument(
        "--repo_ids",
        required=True,
        help="Comma-separated list of repository IDs to combine datasets from."
    )
    parser.add_argument(
        "--combined_repo_id",
        required=True,
        help="The repository ID for the combined dataset."
    )
    parser.add_argument(
        "--use_symlinks",
        action="store_true",
        help="Flag to indicate whether to use symlinks for data and videos."
    )
    return parser.parse_args()

def load_metadata(repo_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata from a repository's info.json file."""
    info_path = os.path.join(HF_LEROBOT_HOME, repo_id, META_DIR, INFO_FILE)
    if not os.path.exists(info_path):
        logging.warning(f"Metadata file {info_path} does not exist. Skipping.")
        return None

    with open(info_path, "r") as f:
        return json.load(f)

def merge_info_files(repo_ids: List[str]) -> Optional[Dict[str, Any]]:
    """Merge info.json files from multiple repositories."""
    combined_info = None

    for repo_id in repo_ids:
        info_data = load_metadata(repo_id)
        if not info_data:
            continue

        if combined_info is None:
            combined_info = info_data.copy()
            combined_info["total_episodes"] = 0
            combined_info["total_frames"] = 0
            combined_info["total_videos"] = 0

        combined_info["total_episodes"] += info_data.get("total_episodes", 0)
        combined_info["total_frames"] += info_data.get("total_frames", 0)
        combined_info["total_videos"] += info_data.get("total_videos", 0)

    return combined_info

def copy_tasks_file(repo_ids: List[str], combined_meta_dir: str) -> bool:
    """Copy the first valid tasks.jsonl file from the repositories."""

    for repo_id in repo_ids:
        tasks_path = os.path.join(HF_LEROBOT_HOME, repo_id, META_DIR, TASKS_FILE)

        if os.path.exists(tasks_path):
            combined_tasks_path = os.path.join(combined_meta_dir, TASKS_FILE)
            with open(tasks_path, "r") as src, open(combined_tasks_path, "w") as dst:
                dst.write(src.read())
            logging.info(f"Copied {tasks_path} to {combined_tasks_path}.")
            return True
        
    logging.warning("No valid tasks.jsonl file found.")
    return False

def save_combined_info_file(combined_info: Dict[str, Any], combined_meta_dir: str) -> None:
    """Save the combined metadata to the specified repository."""
    combined_info_path = os.path.join(combined_meta_dir, INFO_FILE)

    with open(combined_info_path, "w") as f:
        json.dump(combined_info, f, indent=4)

    logging.info(f"Combined info.json saved to {combined_info_path}.")

def copy_info_files(repo_ids: List[str], combined_repo_id: str) -> None:
    combined_info = merge_info_files(repo_ids)

    if combined_info is None:
        logging.error("No valid info.json files found. Exiting.")
        exit(1)

    save_combined_info_file(combined_info, combined_repo_id)

def copy_episodes_files(repo_ids: List[str], combined_meta_dir: str, episodes_file: str) -> None:
    """Combine episodes.jsonl files from multiple repositories into one with reindexed episode_index."""

    combined_episodes_path = os.path.join(combined_meta_dir, episodes_file)
    current_index = 0

    with jsonlines.open(combined_episodes_path, mode='w') as writer:
        for repo_id in repo_ids:
            episodes_path = os.path.join(HF_LEROBOT_HOME, repo_id, META_DIR, episodes_file)
            if not os.path.exists(episodes_path):
                logging.warning(f"Episodes file {episodes_path} does not exist. Skipping.")
                continue

            with jsonlines.open(episodes_path, mode='r') as reader:
                for episode in reader:
                    if "episode_index" in episode:
                        episode["episode_index"] = current_index
                        current_index += 1
                    writer.write(episode)

            logging.info(f"Processed episodes from {episodes_path}.")

    logging.info(f"Combined episodes.jsonl saved to {combined_episodes_path}.")

def copy_metadata(repo_ids: List[str], combined_repo_id: str) -> None:
    """Copy and combine metadata files from multiple repositories."""

    # Create combined repository meta directory
    combined_meta_dir = os.path.join(HF_LEROBOT_HOME, combined_repo_id, META_DIR)
    os.makedirs(combined_meta_dir, exist_ok=True)

    # Copy and combine info.json files
    combined_info = merge_info_files(repo_ids)
    if combined_info is None:
        logging.error("No valid info.json files found. Exiting.")
        exit(1)
    save_combined_info_file(combined_info, combined_meta_dir)

    # Copy tasks.jsonl file
    tasks_copied = copy_tasks_file(repo_ids, combined_meta_dir)
    if not tasks_copied:
        logging.warning("No tasks.jsonl file was copied.")

    # Copy and combine episodes.jsonl and episodes_stats.jsonl files
    copy_episodes_files(repo_ids, combined_meta_dir, EPISODES_FILE)
    copy_episodes_files(repo_ids, combined_meta_dir, EPISODES_STATS_FILE)

def symlink_data(repo_ids: List[str], combined_repo_id: str) -> None:
    """Create symbolic links to video files in the combined repository with sequential filenames."""
    
    combined_data_dir = os.path.join(HF_LEROBOT_HOME, combined_repo_id, DATA_DIR, "chunk-000")

    if os.path.exists(combined_data_dir):
        shutil.rmtree(combined_data_dir)
    os.makedirs(combined_data_dir)

    current_index = 0

    for repo_id in repo_ids:
        data_dir = os.path.join(HF_LEROBOT_HOME, repo_id, DATA_DIR, "chunk-000")
        if not os.path.exists(data_dir):
            logging.warning(f"Data directory {data_dir} does not exist. Skipping.")
            continue

        for data_file in os.listdir(data_dir):
            src_path = os.path.join(data_dir, data_file)

            new_filename = f"episode_{current_index:06d}.parquet"
            dst_path = os.path.join(combined_data_dir, new_filename)

            os.symlink(src_path, dst_path)
            logging.info(f"Created symlink from {src_path} to {dst_path}.")
            current_index += 1

def symlink_videos(repo_ids: List[str], combined_repo_id: str) -> None:
    """Create symbolic links to video files in the combined repository with sequential filenames, accounting for subdirectories."""
    
    combined_video_dir = os.path.join(HF_LEROBOT_HOME, combined_repo_id, VIDEO_DIR, "chunk-000")

    if os.path.exists(combined_video_dir):
        shutil.rmtree(combined_video_dir)
    os.makedirs(combined_video_dir)

    subdir_indices = {}

    for repo_id in repo_ids:
        video_dir = os.path.join(HF_LEROBOT_HOME, repo_id, VIDEO_DIR, "chunk-000")
        if not os.path.exists(video_dir):
            logging.warning(f"Video directory {video_dir} does not exist. Skipping.")
            continue

        for subdir in os.listdir(video_dir):
            subdir_path = os.path.join(video_dir, subdir)

            combined_subdir_path = os.path.join(combined_video_dir, subdir)
            os.makedirs(combined_subdir_path, exist_ok=True)

            if subdir not in subdir_indices:
                subdir_indices[subdir] = 0

            for video_file in os.listdir(subdir_path):
                src_path = os.path.join(subdir_path, video_file)

                new_filename = f"episode_{subdir_indices[subdir]:06d}.mp4"
                dst_path = os.path.join(combined_subdir_path, new_filename)

                os.symlink(src_path, dst_path)
                logging.info(f"Created symlink from {src_path} to {dst_path}.")
                subdir_indices[subdir] += 1

def copy_data(repo_ids: List[str], combined_repo_id: str) -> None:
    """Copy video files to the combined repository with sequential filenames."""
    
    combined_data_dir = os.path.join(HF_LEROBOT_HOME, combined_repo_id, DATA_DIR, "chunk-000")

    if os.path.exists(combined_data_dir):
        shutil.rmtree(combined_data_dir)
    os.makedirs(combined_data_dir)

    current_index = 0

    for repo_id in repo_ids:
        data_dir = os.path.join(HF_LEROBOT_HOME, repo_id, DATA_DIR, "chunk-000")
        if not os.path.exists(data_dir):
            logging.warning(f"Data directory {data_dir} does not exist. Skipping.")
            continue

        for data_file in os.listdir(data_dir):
            src_path = os.path.join(data_dir, data_file)

            new_filename = f"episode_{current_index:06d}.parquet"
            dst_path = os.path.join(combined_data_dir, new_filename)

            shutil.copy(src_path, dst_path)
            logging.info(f"Copied {src_path} to {dst_path}.")
            current_index += 1

def copy_videos(repo_ids: List[str], combined_repo_id: str) -> None:
    """Copy video files to the combined repository with sequential filenames, accounting for subdirectories."""
    
    combined_video_dir = os.path.join(HF_LEROBOT_HOME, combined_repo_id, VIDEO_DIR, "chunk-000")

    if os.path.exists(combined_video_dir):
        shutil.rmtree(combined_video_dir)
    os.makedirs(combined_video_dir)

    subdir_indices = {}

    for repo_id in repo_ids:
        video_dir = os.path.join(HF_LEROBOT_HOME, repo_id, VIDEO_DIR, "chunk-000")
        if not os.path.exists(video_dir):
            logging.warning(f"Video directory {video_dir} does not exist. Skipping.")
            continue

        for subdir in os.listdir(video_dir):
            subdir_path = os.path.join(video_dir, subdir)

            combined_subdir_path = os.path.join(combined_video_dir, subdir)
            os.makedirs(combined_subdir_path, exist_ok=True)

            if subdir not in subdir_indices:
                subdir_indices[subdir] = 0

            for video_file in os.listdir(subdir_path):
                src_path = os.path.join(subdir_path, video_file)

                new_filename = f"episode_{subdir_indices[subdir]:06d}.mp4"
                dst_path = os.path.join(combined_subdir_path, new_filename)

                shutil.copy(src_path, dst_path)
                logging.info(f"Copied {src_path} to {dst_path}.")
                subdir_indices[subdir] += 1

def main() -> None:
    """Main function to combine datasets."""
    args = parse_arguments()
    repo_ids = args.repo_ids.split(",")
    combined_repo_id = args.combined_repo_id

    logging.info(f"Combining datasets from {repo_ids} into {combined_repo_id}.")
    copy_metadata(repo_ids, combined_repo_id)

    if args.use_symlinks:
        symlink_data(repo_ids, combined_repo_id)
        symlink_videos(repo_ids, combined_repo_id)
    else:
        copy_data(repo_ids, combined_repo_id)
        copy_videos(repo_ids, combined_repo_id)


if __name__ == "__main__":
    main()
