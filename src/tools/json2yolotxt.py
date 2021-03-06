####################################################################
#### THIS SCRIPT WAS MODIFIED TO FIT THE CUSTOM NEEDS           ####
#### ORIGINAL SCRIPT: https://github.com/ultralytics/JSON2YOLO  ####
####################################################################

import json
import cv2
import pandas as pd
from PIL import Image
import sys
import os
from tqdm import tqdm
import numpy as np
import shutil
from pathlib import Path
import argparse


# Convert Labelbox JSON file into YOLO-format labels ---------------------------
def convert_labelbox_json(name, file, darknet, filters):
    # Create folders
    path = make_folders()

    # Import json
    with open(file) as f:
        data = json.load(f)

    # Write images and shapes
    name = 'out' + os.sep + name
    file_id, file_name, width, height = [], [], [], []
    for i, x in enumerate(tqdm(data['images'], desc='Files and Shapes')):
        file_id.append(x['id'])
        file_name.append(x['file_name'])
        width.append(x['width'])
        height.append(x['height'])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % file_name[i])

        # shapes
        with open(name + '.shapes', 'a') as file:
            file.write('%g, %g\n' % (x['width'], x['height']))

    # Write *.names file
    categories = []
    for x in tqdm(data['categories'], desc='Names'):
        if x["name"] in filters: continue
        categories.append(x["name"])
        with open(name + '.names', 'a') as file:
            file.write('%s\n' % x['name'])

    # Write labels file
    output_annotations = {}
    with open('out/_annotations.txt', 'a') as file:
        for x in tqdm(data['annotations'], desc='Annotations'):
            i = file_id.index(x['image_id'])  # image index
            label_name = Path(file_name[i]).stem + ('.txt' if darknet else '.jpg')
            if i not in output_annotations:
                output_annotations[i] = label_name

            # The Labelbox bounding box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            if darknet:
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= width[i]  # normalize x
                box[[1, 3]] /= height[i]  # normalize y
            else:
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]

            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                if darknet and len(categories) > int(x['category_id'] - 1) and categories[
                    int(x['category_id'] - 1)] not in filters:
                    with open(f"out/labels/{label_name}", "a") as fileDarknet:
                        fileDarknet.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))
                else:
                    output_annotations[i] += ' %d,%d,%d,%d,%g' % (*box, x['category_id'] - 1)
        if not darknet:
            for line in output_annotations.values():
                file.write(line + "\n")

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


def make_folders(path='out/'):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path


def split_files(out_path, file_name, prefix_path=''):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=0.9, test=0.1, validate=0.0)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(out_path + '_' + key + '.txt', 'w+') as file:
                for i in item:
                    file.write('%s%s\n' % (prefix_path, file_name[i]))


def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert COCOJSON format to yolo txt format.")
    parser.add_argument("-name", required=True, type=str, help="file name")
    parser.add_argument("-json", required=True, type=str, help="json file to convert")
    parser.add_argument("-darknet", type=bool, help="should the txt files be compatible with the darknet format?")
    parser.add_argument("-filter-categories", type=str, nargs="*", help="Filter categories in dataset. Expects indexes")

    args = parser.parse_args()

    convert_labelbox_json(name=args.name, file=args.json, darknet=args.darknet, filters=args.filter_categories)
