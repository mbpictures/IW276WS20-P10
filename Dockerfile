FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

# Needed for accessing Jetpack 4.4
COPY  /docker-requirements/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  /docker-requirements/jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

# Clone the P10-repo master
#RUN cd home && \
#        git clone https://github.com/IW276/IW276WS20-P10.git
WORKDIR /home/IW276WS20-P10
COPY . .

RUN pip3 install -U \
        pip \
        setuptools \
        wheel && \
    pip3 install \
        -r requirements.txt \
         && \
    rm -rf ~/.cache/pip

#RUN cd /home && \
#        git clone https://github.com/jkjung-avt/tensorrt_demos.git

WORKDIR /home/IW276WS20-P10/demos/tensorrt_demos

RUN pip3 install -U protobuf

RUN apt-get update && \ 
    apt-get install -y cmake libopencv-python libboost-python-dev libboost-thread-dev && \
    apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          build-essential \
          zlib1g-dev \
          zip \
          libjpeg8-dev && rm -rf /var/lib/apt/lists/*

# TODO: Copy pycuda, instead of installing it 
RUN cd ssd && \
    ./install_pycuda.sh

# TODO: Fix onnx error
RUN pip3 install onnx==1.4.1

RUN cd plugins && \ 
   make

WORKDIR /home/

# ....
