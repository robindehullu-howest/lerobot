#!/bin/bash
set -e

if [ -n "$SA_KEY" ]; then
    echo "Writing SA_KEY to $GOOGLE_APPLICATION_CREDENTIALS..."
    mkdir -p "$(dirname "$GOOGLE_APPLICATION_CREDENTIALS")"
    echo "$SA_KEY" > "$GOOGLE_APPLICATION_CREDENTIALS"
    chmod 600 "$GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "SA_KEY is not set. Exiting."
    exit 1
fi    

echo "Checking for W&B API key..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into W&B..."
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set. Skipping W&B login."
fi

if [ -z "$DATASET_IDS" ]; then
    echo "DATASET_IDS is not set. Exiting."
    exit 1
fi

echo "Downloading dataset..."
python lerobot/common/datasets/gcs_utils.py \
    --bucket_name=robot-445714_lerobot_train_data \
    --action=pull --content_type=dataset \
    --identifiers="${DATASET_IDS}" \

echo "Combining dataset..."
python lerobot/common/datasets/combine_datasets.py \
    --repo_ids="${DATASET_IDS}" \
    --combined_repo_id so100/training_set \
    --excluded_video_keys="${EXCLUDED_CAMERAS}"

echo "Dataset preparation complete."

exec "$@"