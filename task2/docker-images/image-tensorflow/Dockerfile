# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #==========================================================================
FROM tensorflow/tensorflow:1.12.0-gpu-py3
#this was done 3/1/2019

# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get update && \
    apt-get install -y wget vim emacs nano git

# Install critical packages
RUN apt-get update && \
    apt-get install -y protobuf-compiler python-pil python-lxml python-tk python-setuptools python3-pip && \
    pip3 install --upgrade numpy \
      tensorflow-gpu==1.12.0 \
      Cython \
      contextlib2 \
      jupyter \
      matplotlib \
      Keras \
      Keras-Applications \
      Keras-Preprocessing \
      sklearn \
      scikit-image \
      scikit-learn \
      scipy \
      pandas \
      opencv-python

# Install additional pip packages
RUN pip3 install --upgrade \
      shapely \
      progressbar2 \
      opencv-contrib-python

WORKDIR /data/datasets/tscherli
