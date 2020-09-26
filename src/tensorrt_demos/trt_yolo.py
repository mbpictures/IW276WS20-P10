"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time
import argparse
import json
import numpy
import cv2

import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME = 'TrtYOLODemo'
ACTIVATE_DISPLAY = False
WRITE_IMAGES = False
validCocoJson = dict()
resultJson = list()
IMAGE_OUTPUT = "/home/out/images"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-v', '--valid_coco', type=str,
        help='Path to the valid coco json file')
    parser.add_argument(
        '--write_images', action="store_true",
        help='Write images with detected bounding boxes to output directory')
    args = parser.parse_args()
    return args


def process_valid_json(coco_json_file):
    global validCocoJson
    with open(coco_json_file, "r") as jsonFile:
        data = jsonFile.read()
        result = json.loads(data)
    for image in result["images"]:
        validCocoJson[os.path.basename(image["file_name"])] = image["id"]


def append_coco(boxes, confidences, classes, camera):
    global resultJson
    global validCocoJson

    for box, confidence, classes in zip(boxes, confidences, classes):
        class_id = int(classes)
        annotation = {
            'image_id': validCocoJson[os.path.basename(camera.currentImage)],
            'bbox': [
                int(box[0]),
                int(box[1]),
                int(box[2] - box[0]),
                int(box[3] - box[1])
            ],
            'score': confidence,
            'category_id': class_id
        }
        resultJson.append(annotation)


def loop_and_detect(camera, trt_yolo, args, confidence_thresh, visual):
    """Continuously capture images from camera and do object detection.

    # Arguments
      camera: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      confidence_thresh: confidence/score threshold for object detection.
      visual: for visualization.
    """
    fps = 0.0

    while len(camera.imageNames) != 0:
        if ACTIVATE_DISPLAY and (cv2.getWindowProperty(WINDOW_NAME, 0) < 0):
            break
        img = camera.read()
        if img is None:
            break
        tic = time.time()
        boxes, confidences, classes = trt_yolo.detect(img, confidence_thresh)
        toc = time.time()
        if ACTIVATE_DISPLAY or args.write_images:
            img = visual.draw_bboxes(img, boxes, confidences, classes)
            img = show_fps(img, fps)
        if ACTIVATE_DISPLAY:
            cv2.imshow(WINDOW_NAME, img)
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        if ACTIVATE_DISPLAY:
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
        if args.write_images:
            path = os.path.join(IMAGE_OUTPUT, os.path.basename(camera.currentImage))
            print("Image path: ", path)
            cv2.imwrite(path, img)
        print("FPS: {:3.2f} and {} Images left.".format(fps, len(camera.imageNames)))
        append_coco(boxes, confidences, classes, camera)

    # Write coco json file when done
    coco_file = json.dumps(resultJson, cls=NpEncoder)
    f = open("/home/out/result.json", "w+")
    f.write(coco_file)
    f.close()


def main():
    global cocoCategoryId
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit(f'ERROR: bad category_num ({args.category_num})!')
    if not os.path.isfile(args.model):
        raise SystemExit(f'ERROR: file {args.model} not found!')

    # Process valid coco json file
    process_valid_json(args.valid_coco)

    if args.write_images:
        if not os.path.exists(IMAGE_OUTPUT): os.mkdir(IMAGE_OUTPUT)

    # Create camera for video/image input
    cam = Camera(args)
    if not cam.get_is_opened():
        raise SystemExit('ERROR: failed to open camera!')

    class_dict = get_cls_dict(args.category_num)
    yolo_dim = (args.model.replace(".trt", "")).split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit(f'ERROR: bad yolo_dim ({yolo_dim})!')
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit(f'ERROR: bad yolo_dim ({yolo_dim})!')

    # Create yolo
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    if ACTIVATE_DISPLAY:
        open_window(
            WINDOW_NAME, 'Camera TensorRT YOLO Demo',
            cam.img_width, cam.img_height)
    visual = BBoxVisualization(class_dict)

    # Run detection
    loop_and_detect(cam, trt_yolo, args, confidence_thresh=0.3, visual=visual)

    # Clean up
    cam.release()
    if ACTIVATE_DISPLAY:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
