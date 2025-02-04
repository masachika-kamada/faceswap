FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    wget \
    vim \
    python3.8 \
    python3.8-distutils \
    ffmpeg \
    libsm6 \
    libxext6 \
    gcc \
    libquadmath0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.8 get-pip.py

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80
ENV PORT 80

ENTRYPOINT []
CMD ["python3.8", "/workspace/server.py"]

WORKDIR /home/ubuntu