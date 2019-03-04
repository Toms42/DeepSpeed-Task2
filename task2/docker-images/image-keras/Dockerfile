FROM nvidia/cuda:9.0-cudnn7-devel
ARG cuda_version=9.0
ARG cudnn_version=7

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      protobuf-compiler \
      python-pil \
      python-lxml \
      python-tk \
      python-dev \
      python-setuptools \
      python3-setuptools \
      python-pip \
      python3-pip \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

RUN pip install --upgrade pip && \
    pip3 install \
      sklearn_pandas \
      tensorflow-gpu==1.12.0 \
      Cython \
      contextlib2 \
      scipy \
      shapely \
      progressbar2 \
      opencv-python \
      opencv-contrib-python \
      h5py
RUN pip3 install \
      matplotlib \
      mkl \
      nose \
      Pillow \
      pandas \
      pydot \
      pyyaVml
RUN pip3 install \
      scikit-learn \
      scikit-image \
      six \
      mkdocs \
      cntk-gpu
RUN git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /data/datasets/tscherli

