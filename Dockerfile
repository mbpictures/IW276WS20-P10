FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

# Needed for accessing Jetpack 4.4
COPY  /docker-requirements/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  /docker-requirements/jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN apt-get update && \ 
    apt-get install -y libopencv-python libboost-python-dev libboost-thread-dev && \
    apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          build-essential \
          zlib1g-dev \
          zip \
          libjpeg8-dev && \ 
    rm -rf /var/lib/apt/lists/*

# Do not change order!
RUN pip3 install -U setuptools Cython wheel 
RUN pip3 install numpy

RUN apt-get update && \ 
    apt-get install -y cmake libprotoc-dev libprotobuf-dev protobuf-compiler

RUN pip3 install -U \
        pip \
        setuptools \
        Cython \ 
        wheel \
        protobuf \ 
        onnx==1.4.1

# Copy IW276WS20-P10 into docker
WORKDIR /home/IW276WS20-P10
COPY . .

WORKDIR /home/IW276WS20-P10/src/tensorrt_demos

# Install pycuda 
RUN cd ssd && \
    ./install_pycuda.sh

# Should already be built. Just for 
RUN cd plugins && \ 
   make

WORKDIR /home/
RUN mkdir out
RUN mkdir in
WORKDIR /home/IW276WS20-P10/src/tensorrt_demos

# ....
