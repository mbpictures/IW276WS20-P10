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

> sudo python3 build_docker.py
3. Start the docker container (run "python3 start_docker.py -h" for help)
> sudo python3 start_docker.py --input "Directory where the images to detect are stored" --output "Directory where the output is stored" --image "name and tag of the docker container to run" --valid-json "Path to the valid json file" [--tiny] [--write_images]


## Docker
HOW TO

## Acknowledgments

This repo is based on
  - [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)
  - [PANDA-Toolkit](https://github.com/GigaVision/PANDA-Toolkit)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
