# --------------------------------------------------------
# Compute metrics for detectors using ground-truth data
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Based on pycocotools (https://github.com/cocodataset/cocoapi/)
# --------------------------------------------------------

#### MODIFIED VERSION #####
import json
import numpy as np
import argparse

CATEGORY = {
    'visible body': 1,
    'full body': 2,
    'head': 3,
    'vehicle': 4
}

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for detectors using ground-truth data.

Files
-----
All result files have to comply with the COCO format described in
http://cocodataset.org/#format-results
Structure
---------

Layout for ground truth data
    <GT_ROOT>/anno.json'

Layout for test data
    <TEST_ROOT>/results.json
""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('personfile', type=str, help='File path to person annotation json file')
    parser.add_argument('transferred', type=str, help='Directory containing transferred gt files', default='transferred.json')
    parser.add_argument('--annType', type=str, help='annotation type', default='bbox')
    parser.add_argument('--maxDets', type=list, help='[10, 100, 500] M = 3 thresholds on max detections per image',
                        default=[10, 100, 500])
    parser.add_argument('--areaRng', type=list, help='[...] A = 4 object area ranges for evaluation',
                        default=[0, 200, 400, 1e5])
    return parser.parse_args()

def generate_coco_anno(personsrcfile, tgtfile, keywords=None):
    """
    transfer ground truth to COCO format
    :param personsrcfile: person ground truth file path
    :param vehiclesrcfile: vehicle ground truth file path
    :param tgtfile: generated file save path
    :param keywords: list of str, only keep image with keyword in image name
    :return:
    """
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "none", "id": 1, "name": 'visible body'},
        {"supercategory": "none", "id": 2, "name": 'full body'},
        {"supercategory": "none", "id": 3, "name": 'head'},
        {"supercategory": "none", "id": 4, "name": 'vehicle'}
    ]
    with open(personsrcfile, 'r') as load_f:
        person_anno_dict = json.load(load_f)

    images = list()
    annotations = list()
    imageids = list()

    objid = 1
    for (imagename, imagedict) in person_anno_dict.items():
        if keywords:
            flag = False
            for kw in keywords:
                if kw in imagename:
                    flag = True
            if not flag:
                continue
        image = dict()
        image['file_name'] = imagename
        imgid = imagedict['image id']
        imageids.append(imgid)
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        image['height'] = imgheight
        image['width'] = imgwidth
        image['id'] = imgid
        images.append(image)
        for objdict in imagedict['objects list']:
            cate = objdict['category']
            if cate == 'person':
                for label in ['visible body', 'full body', 'head']:
                    rect = objdict['rects'][label]
                    annotation = dict()
                    annotation["image_id"] = imgid
                    annotation["ignore"] = 0
                    annotation["iscrowd"] = 0
                    x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                    annotation["bbox"] = [x, y, w, h]
                    annotation["area"] = float(w * h)
                    annotation["category_id"] = CATEGORY[label]
                    annotation["id"] = objid
                    objid += 1
                    annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                    annotations.append(annotation)
            else:
                annotation = dict()
                if cate == 'crowd':
                    annotation["iscrowd"] = 1
                else:
                    annotation["iscrowd"] = 0
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["ignore"] = 1
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['visible body']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)

        # ADDITIONAL: convert vehicle annos as well, but as we don't need them, this code isn't necessary anymore

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

    print("Converted JSON successfully")
    return imageids

def RectDict2List(rectdict, imgwidth, imgheight, scale, mode='tlbr'):
    x1, y1, x2, y2 = restrain_between_0_1([rectdict['tl']['x'], rectdict['tl']['y'],
                                           rectdict['br']['x'], rectdict['br']['y']])
    xmin = int(x1 * imgwidth * scale)
    ymin = int(y1 * imgheight * scale)
    xmax = int(x2 * imgwidth * scale)
    ymax = int(y2 * imgheight * scale)

    if mode == 'tlbr':
        return xmin, ymin, xmax, ymax
    elif mode == 'tlwh':
        return xmin, ymin, xmax - xmin, ymax - ymin

def restrain_between_0_1(values_list):
    return_list = []
    for value in values_list:
        if value < 0:
            new_value = 0
        elif value > 1:
            new_value = 1
        else:
            new_value = value
        return_list.append(new_value)

    return return_list


if __name__ == '__main__':
    args = parse_args()

    # transfer ground truth to COCO format
    generate_coco_anno(args.personfile, args.transferred)
