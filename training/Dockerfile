# syntax=docker/dockerfile:1
# GPU training image for semaje voices.
#   docker build -f training/Dockerfile -t semaje-tts-train .
#   docker run --gpus all -v $PWD/work:/app/work semaje-tts-train make voice LANG=kik ACCEL=gpu
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 SEMAJE_HOME=/opt/semaje \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 WAXAL_CACHE=/app/work/.cache/waxal PIPER_BASE_CACHE=/app/work/.cache/piper-base
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev build-essential curl ffmpeg make espeak-ng \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
COPY training/ training/
RUN PYTHON=python3.11 training/setup.sh
COPY semaje_tts/ semaje_tts/
COPY Makefile ./
CMD ["make", "voice", "ACCEL=gpu"]
