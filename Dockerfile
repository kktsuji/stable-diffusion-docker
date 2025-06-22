FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3.10-venv git
