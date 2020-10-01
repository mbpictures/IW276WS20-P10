# Person Detection Challenge

With the increasing number of CCTV in public spaces it is of interest to provide automatic analysis solutions in terms of e.g. image detection to provide safety precautions which fulfill both privacy damands and reliable results.
This solution utilizes a python tensorRT pipeline to analyze the video signal of a camera (a series of pictures) on a nvidia jetson nano embedded device and focuses on detecting large numbers of people.

<p align="center">
  <img src="doc/img/undetected.gif" width="861" />
  <img src="doc/img/detected.gif" width="861" />
  
</p>
TODO: Insert Link to Demo Video

> This work was done by Marius Butz, Tim Hänlein and Valeria Zitz during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-models)
* [Running](#running)
* [Acknowledgments](#acknowledgments)

## Requirements
### Required
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* Jetson Nano
* Jetpack 4.4
* make
* cmake
* g++
### Optional
* Docker
* dtrx

## Prerequisites
1. Install requirements:
```bash
pip install -r src/tools/requirements.txt
```
```bash
pip install -r src/tensorrt_demos/requirements.txt
```

## Pre-trained models <a name="pre-trained-models"/>

Two pre-trained models are available at pretrained-models/ :

custom-yolov4-tiny-detector_final-416x416.trt:
* Is a slimmed-down YoloV4 model in tensorRT format. It it way light weighter and faster than the YoloV4 model at the cost of detection accuracy.

custom-yolov4-detector_final-416x416.trt:
* Is a full-fledged YoloV4 model. It is slower than the tiny model, but provides more accurate results in terms of both confidence and bounding box dimensions.
To unzip the file execute the following commands.
```bash
cd pretrained-models/
```
```bash
sudo dtrx custom-yolov4-detector_final-416x416.zip -v -n -f
```

Note that this step is not required for building the docker image. Docker will unzip this file itself.

## Running
### Docker
1. Clone the repository
```bash
git clone https://github.com/IW276/IW276WS20-P10.git
```
2. Build the docker image
```bash
cd IW276WS20-P10/
```
```bash
sudo python3 build_docker.py
```
3. Start the docker container (run "python3 start_docker.py -h" for help)
```bash
sudo python3 start_docker.py --input "Directory where the images to detect are stored" --output "Directory where the output is stored" --image "name and tag of the docker container to run" --valid-json "Path to the valid json file" [--tiny] [--write_images]
```
### Python
To run the prediction without docker, please execute the following commands.
```bash
cd src/tensorrt_demos/yolo
```
```bash
sudo python3 trt_yolo.py --imageDir "directory containing images" -v "path to ground truth coco json" -m "path to tensorrt model" [--write_images] [--image_output "directory where the images with bounding boxes should be stored (only when --write_images is enabled)" Default: "/home/out/images"] [--result_json "path to the json, which contains all detected annotations" Default: "/home/out/result.json"] [--confidence_threshhold Default: 0.3] [--activate_display]
```


## Acknowledgments

This repo is based on
  - [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)
  - [PANDA-Toolkit](https://github.com/GigaVision/PANDA-Toolkit)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
