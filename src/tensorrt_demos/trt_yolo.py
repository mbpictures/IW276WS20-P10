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
cocoAnnotationId = 1
cocoImageId = 0
cocoCategoryId = 0
cocoJson = dict()


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
    args = parser.parse_args()
    return args


def append_coco(boxes, confidences, classes, camera):
    global cocoImageId
    global cocoAnnotationId
    global cocoJson
    cocoImage = { 
        'file_name':camera.currentImage,
        'height':camera.img_height,
        'width':camera.img_width,
        'id':cocoImageId
    }

    for box, confidence, clss in zip(boxes, confidences, classes):
        classId = int(clss)
        annotation = {
            'id':cocoAnnotationId,
            'bbox':[
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3])
            ] , 
            'image_id':cocoImageId, 
            'segmentation':[], 
            'ignore':0, 
            'area':(box[2]-box[0]) * (box[3]-box[1]), 
            'iscrowd':0, 
            'category_id':classId
        }
        cocoJson['annotations'].append(annotation)
        cocoAnnotationId += 1

    cocoJson['images'].append(cocoImage)
    cocoImageId += 1


def loop_and_detect(camera, trt_yolo, args, confidence_thresh, visual):
    """Continuously capture images from camera and do object detection.

    # Arguments
      camera: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      confidence_thresh: confidence/score threshold for object detection.
      visual: for visualization.
    """
    fps = 0.0
    tic = time.time()
    imageWritten = False

    while len(camera.imageNames) != 0:
        if ACTIVATE_DISPLAY:
            if (cv2.getWindowProperty(WINDOW_NAME, 0) < 0):
                break
        img = camera.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, confidence_thresh)
        img = visual.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        if ACTIVATE_DISPLAY:
            cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        if ACTIVATE_DISPLAY:
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
        print("FPS: {:3.2f} and {} Images left.".format(fps, len(camera.imageNames)))
        #if not imageWritten:
        #    cv2.imwrite("/home/Pictures/test.png", img)
        #    imageWritten = True
        append_coco(boxes, confs, clss, camera)
    
    # Write coco json file when done
    cocoFile = json.dumps(cocoJson, cls=NpEncoder)
    f = open("/home/out/cocoFile.json", "w+")
    f.write(cocoFile)
    f.close()
            

def main():
    global cocoCategoryId
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    # Create camera for video/image input
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    # Create yolo
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    if ACTIVATE_DISPLAY:
        open_window(
            WINDOW_NAME, 'Camera TensorRT YOLO Demo',
            cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)

    # Add single occuring tags to coco-json
    cocoJson['type'] = "instances"
    cocoJson["categories"] = list()
    for category in cls_dict:
        cocoJson["categories"].append({
            'supercategory':'none',
            'name':cls_dict.get(category, 'CLS{}'.format(category)),
            'id':cocoCategoryId
        })
        cocoCategoryId += 1
    cocoJson["images"] = list()
    cocoJson["annotations"] = list()

    # Run detection
    loop_and_detect(cam, trt_yolo, args, confidence_thresh=0.3, visual=vis)

    # Clean up
    cam.release()
    if ACTIVATE_DISPLAY:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
