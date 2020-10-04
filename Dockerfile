FROM ubuntu:18.04
FROM python:3.7
FROM nvidia/cuda:10.1-base-ubuntu18.04
FROM tensorflow/tensorflow:2.0.1

LABEL maintainer "siva balasubramanian <sivabala94@gmail.com>"
LABEL Description "VAE TesorFlow docker."

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    rsync \
    vim \
    sox \
    libsndfile1

RUN pip install tqdm==4.43.0 librosa==0.7.2 opencv-python==4.2.0.32 SoundFile==0.10.3.post1 PyYAML==5.3

# COPY train.py /opt/tf/train.py

# RUN pip install --upgrade pip
# RUN pip install wheel
# RUN pip install sagemaker-containers

# WORKDIR /opt/tf
WORKDIR /home/ubuntu/TensorFlow

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

ENTRYPOINT bash

# ENV SAGEMAKER_PROGRAM train.py
