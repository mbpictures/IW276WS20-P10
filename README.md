# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Marius Butz, Tim Hänlein and Valeria Zitz during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-models)
* [Running](#running)
* [Acknowledgments](#acknowledgments)

## Requirements
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* Jetson Nano
* Jetpack 4.4
> [Optional] ...

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```

## Pre-trained models <a name="pre-trained-models"/>

Pre-trained model is available at pretrained-models/

## Running
1. Clone the repository
> git clone https://github.com/IW276/IW276WS20-P10.git
2. Build the docker image
> cd IW276WS20-P10/

> sudo ./build_docker_image
3. Start the docker container (consider passing a mounting directory for images/videos with -v source:destination)
> sudo docker run -it --rm --runtime nvidia --network host iw276ws20-p10:0.1
4. Start the TensorRT demo
> cd IW276WS20-P10/src/tensorrt_demos/

> python3 trt_yolo.py --image /home/Pictures/some_picture.jpg -m yolov4-416

or

> python3 trt_yolo.py --video /home/Videos/some_video.mp4 -m yolov4-416

## Docker
HOW TO

## Acknowledgments

This repo is based on
  - [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
