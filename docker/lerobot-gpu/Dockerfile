FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"
ENV GOOGLE_APPLICATION_CREDENTIALS="/.keys/service-account-key.json"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && python -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Clone repository and install LeRobot in a single layer
COPY . /lerobot
WORKDIR /lerobot
RUN chmod +x /lerobot/docker/lerobot-gpu/entrypoint.sh && \
    /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
    && /opt/venv/bin/pip install --no-cache-dir ".[google_cloud]" 

ENTRYPOINT [ "/lerobot/docker/lerobot-gpu/entrypoint.sh" ]
CMD ["sh", "-c", "python lerobot/scripts/train.py \
    --dataset.repo_id so100/training_set \
    --policy.type ${POLICY} \
    --output_dir outputs/train/${MODEL_ID} \
    --job_name ${JOB_NAME} \
    --wandb.enable true \
    --batch_size ${BATCH_SIZE} \
    --steps ${MAX_STEPS} \
    --save_freq ${SAVE_STEPS} \
    --gcs_bucket ${MODEL_BUCKET_NAME} \
    --num_workers ${NUM_DATALOADERS}"]