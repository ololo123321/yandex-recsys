FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip curl vim
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt