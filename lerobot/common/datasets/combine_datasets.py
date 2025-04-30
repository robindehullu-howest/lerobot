import logging
import argparse
import os
import json
import shutil
import jsonlines
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

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
        "--base_dir",
        default=Path(os.getenv("HOME") or os.getenv("USERPROFILE"), ".cache", "huggingface", "lerobot"),
        help="Base directory for the dataset."
    )
    parser.add_argument(
        "--use_symlinks",
        action="store_true",
        help="Flag to indicate whether to use symlinks for data and videos."
    )
    return parser.parse_args()

def load_metadata(repo_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metadata from a repository's info.json file."""
    info_path = Path(repo_dir, META_DIR, INFO_FILE)
    if not os.path.exists(info_path):
        logging.warning(f"Metadata file {info_path} does not exist. Skipping.")
        return None

    with open(info_path, "r") as f:
        return json.load(f)

def merge_info_files(repo_ids: List[str], base_dir: str) -> Optional[Dict[str, Any]]:
    """Merge info.json files from multiple repositories."""
    combined_info = None

    for repo_id in repo_ids:
        info_data = load_metadata(Path(base_dir, repo_id))
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

        if "splits" in combined_info and "train" in combined_info["splits"]:
            combined_info["splits"]["train"] = f"0:{combined_info['total_episodes']}"
            
    return combined_info

def copy_tasks_file(repo_ids: List[str], base_dir: str, combined_meta_dir: str) -> bool:
    """Copy the first valid tasks.jsonl file from the repositories."""

    for repo_id in repo_ids:
        tasks_path = Path(base_dir, repo_id, META_DIR, TASKS_FILE)

        if os.path.exists(tasks_path):
            combined_tasks_path = Path(combined_meta_dir, TASKS_FILE)
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

def copy_episodes_files(repo_ids: List[str], combined_meta_dir: str, episodes_file: str, base_dir: str) -> None:
    """Combine episodes.jsonl files from multiple repositories into one with reindexed episode_index."""

    combined_episodes_path = Path(combined_meta_dir, episodes_file)
    current_index = 0

    with jsonlines.open(combined_episodes_path, mode='w') as writer:
        for repo_id in repo_ids:
            episodes_path = Path(base_dir, repo_id, META_DIR, episodes_file)
            if not episodes_path.exists():
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

def copy_metadata(repo_ids: List[str], combined_repo_id: str, base_dir: str) -> None:
    """Copy and combine metadata files from multiple repositories."""

    # Create combined repository meta directory
    combined_meta_dir = Path(base_dir, combined_repo_id, META_DIR)
    os.makedirs(combined_meta_dir, exist_ok=True)

    # Copy and combine info.json files
    combined_info = merge_info_files(repo_ids, base_dir)
    if combined_info is None:
        logging.error("No valid info.json files found. Exiting.")
        exit(1)
    save_combined_info_file(combined_info, combined_meta_dir)

    # Copy tasks.jsonl file
    tasks_copied = copy_tasks_file(repo_ids, base_dir, combined_meta_dir)
    if not tasks_copied:
        logging.warning("No tasks.jsonl file was copied.")

    # Copy and combine episodes.jsonl and episodes_stats.jsonl files
    copy_episodes_files(repo_ids, combined_meta_dir, EPISODES_FILE, base_dir)
    copy_episodes_files(repo_ids, combined_meta_dir, EPISODES_STATS_FILE, base_dir)

def copy_data(repo_ids: List[str], combined_repo_id: str, base_dir: str) -> None:
    """Copy and combine parquet files to the combined repository with updated indices."""
    
    combined_data_dir = Path(base_dir, combined_repo_id, DATA_DIR, "chunk-000")

    if combined_data_dir.exists():
        shutil.rmtree(combined_data_dir)
    combined_data_dir.mkdir(parents=True)

    current_episode_index = 0
    current_frame_index = 0

    for repo_id in repo_ids:
        data_dir = Path(base_dir, repo_id, DATA_DIR, "chunk-000")
        if not data_dir.exists():
            logging.warning(f"Data directory {data_dir} does not exist. Skipping.")
            continue

        for data_file in sorted(data_dir.iterdir()):
            src_path = Path(data_dir, data_file)

            df = pd.read_parquet(src_path)
            df["episode_index"] = current_episode_index
            df["index"] = range(current_frame_index, current_frame_index + len(df))

            new_filename = f"episode_{current_episode_index:06d}.parquet"
            dst_path = os.path.join(combined_data_dir, new_filename)
            df.to_parquet(dst_path, index=False)

            current_episode_index += 1
            current_frame_index += len(df)

            logging.info(f"Processed and copied {src_path} to {dst_path}.")

def copy_videos(repo_ids: List[str], combined_repo_id: str, base_dir:str) -> None:
    """Copy video files to the combined repository with sequential filenames, accounting for subdirectories."""
    
    combined_video_dir = Path(base_dir, combined_repo_id, VIDEO_DIR, "chunk-000")

    if combined_video_dir.exists():
        shutil.rmtree(combined_video_dir)
    combined_video_dir.mkdir(parents=True)

    subdir_indices = {}

    for repo_id in repo_ids:
        video_dir = Path(base_dir, repo_id, VIDEO_DIR, "chunk-000")
        if not video_dir.exists():
            logging.warning(f"Video directory {video_dir} does not exist. Skipping.")
            continue

        for subdir in sorted(video_dir.iterdir()):
            subdir_path = Path(video_dir, subdir)

            combined_subdir_path = Path(combined_video_dir, subdir)
            combined_subdir_path.mkdir(exist_ok=True)

            if subdir not in subdir_indices:
                subdir_indices[subdir] = 0

            for video_file in sorted(os.listdir(subdir_path)):
                src_path = Path(subdir_path, video_file)

                new_filename = f"episode_{subdir_indices[subdir]:06d}.mp4"
                dst_path = Path(combined_subdir_path, new_filename)

                shutil.copy(src_path, dst_path)
                logging.info(f"Copied {src_path} to {dst_path}.")
                subdir_indices[subdir] += 1

def symlink_videos(repo_ids: List[str], combined_repo_id: str, base_dir: str) -> None:
    """Create symbolic links to video files in the combined repository with sequential filenames, accounting for subdirectories."""
    
    combined_video_dir = Path(base_dir, combined_repo_id, VIDEO_DIR, "chunk-000")

    if combined_video_dir.exists():
        shutil.rmtree(combined_video_dir)
    combined_video_dir.mkdir(parents=True)

    subdir_indices = {}

    for repo_id in repo_ids:
        video_dir = Path(base_dir, repo_id, VIDEO_DIR, "chunk-000")
        if not video_dir.exists():
            logging.warning(f"Video directory {video_dir} does not exist. Skipping.")
            continue

        for subdir in sorted(video_dir.iterdir()):
            subdir_path = Path(video_dir, subdir)

            combined_subdir_path = Path(combined_video_dir, subdir)
            os.makedirs(combined_subdir_path, exist_ok=True)

            if subdir not in subdir_indices:
                subdir_indices[subdir] = 0

            for video_file in sorted(subdir_path.iterdir()):
                src_path = Path(subdir_path, video_file)

                new_filename = f"episode_{subdir_indices[subdir]:06d}.mp4"
                dst_path = Path(combined_subdir_path, new_filename)

                os.symlink(src_path, dst_path)
                logging.info(f"Created symlink from {src_path} to {dst_path}.")
                subdir_indices[subdir] += 1

def main() -> None:
    """Main function to combine datasets."""
    args = parse_arguments()
    repo_ids = args.repo_ids.split(",")
    combined_repo_id = args.combined_repo_id
    base_dir = args.base_dir

    logging.info(f"Combining datasets from {repo_ids} into {combined_repo_id}.")
    copy_metadata(repo_ids, combined_repo_id, base_dir)
    copy_data(repo_ids, combined_repo_id, base_dir)

    if args.use_symlinks:
        symlink_videos(repo_ids, combined_repo_id, base_dir)
    else:
        copy_videos(repo_ids, combined_repo_id, base_dir)


if __name__ == "__main__":
    main()
