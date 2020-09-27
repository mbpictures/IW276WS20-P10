# --------------------------------------------------------
# Compute metrics for detectors using ground-truth data
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Based on pycocotools (https://github.com/cocodataset/cocoapi/)
# --------------------------------------------------------

#### MODIFIED VERSION #####
import json
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
    parser.add_argument('transferred', type=str, help='Directory containing transferred gt files',
                        default='transferred.json')
    parser.add_argument('--annType', type=str, help='annotation type', default='bbox')
    parser.add_argument('--maxDets', type=list, help='[10, 100, 500] M = 3 thresholds on max detections per image',
                        default=[10, 100, 500])
    parser.add_argument('--areaRng', type=list, help='[...] A = 4 object area ranges for evaluation',
                        default=[0, 200, 400, 1e5])
    return parser.parse_args()


def generate_coco_annotation(person_src_file, tgt_file, keywords=None):
    """
    transfer ground truth to COCO format
    :param person_src_file: person ground truth file path
    :param tgt_file: generated file save path
    :param keywords: list of str, only keep image with keyword in image name
    :return:
    """
    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": 'visible body'},
        {"supercategory": "none", "id": 2, "name": 'full body'},
        {"supercategory": "none", "id": 3, "name": 'head'},
        {"supercategory": "none", "id": 4, "name": 'vehicle'}
    ]
    with open(person_src_file, 'r') as load_f:
        person_annotations_dict = json.load(load_f)

    images = list()
    annotations = list()
    image_ids = list()

    object_id = 1
    for (image_name, image_dict) in person_annotations_dict.items():
        if keywords:
            flag = False
            for kw in keywords:
                if kw in image_name:
                    flag = True
            if not flag:
                continue
        image = dict()
        image['file_name'] = image_name
        img_id = image_dict['image id']
        image_ids.append(img_id)
        image_width = image_dict['image size']['width']
        image_height = image_dict['image size']['height']
        image['height'] = image_height
        image['width'] = image_width
        image['id'] = img_id
        images.append(image)
        for object_dict in image_dict['objects list']:
            cate = object_dict['category']
            if cate == 'person':
                for label in ['visible body', 'full body', 'head']:
                    rect = object_dict['rects'][label]
                    annotation = dict()
                    annotation["image_id"] = img_id
                    annotation["ignore"] = 0
                    annotation["iscrowd"] = 0
                    x, y, w, h = rect_dict_2_list(rect, image_width, image_height, scale=1, mode='tlwh')
                    annotation["bbox"] = [x, y, w, h]
                    annotation["area"] = float(w * h)
                    annotation["category_id"] = CATEGORY[label]
                    annotation["id"] = object_id
                    object_id += 1
                    annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                    annotations.append(annotation)
            else:
                annotation = dict()
                if cate == 'crowd':
                    annotation["iscrowd"] = 1
                else:
                    annotation["iscrowd"] = 0
                rect = object_dict['rect']
                annotation["image_id"] = img_id
                annotation["ignore"] = 1
                x, y, w, h = rect_dict_2_list(rect, image_width, image_height, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['visible body']
                annotation["id"] = object_id
                object_id += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)

        # ADDITIONAL: convert vehicle annotations as well, but as we don't need them, this code isn't necessary anymore

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    # print attr_dict
    json_string = json.dumps(attr_dict, indent=2)
    with open(tgt_file, "w") as f:
        f.write(json_string)

    print("Converted JSON successfully")
    return image_ids


def rect_dict_2_list(rect_dict, image_width, image_height, scale, mode='tlbr'):
    x1, y1, x2, y2 = restrain_between_0_1([rect_dict['tl']['x'], rect_dict['tl']['y'],
                                           rect_dict['br']['x'], rect_dict['br']['y']])
    x_min = int(x1 * image_width * scale)
    y_min = int(y1 * image_height * scale)
    x_max = int(x2 * image_width * scale)
    y_max = int(y2 * image_height * scale)

    if mode == 'tlbr':
        return x_min, y_min, x_max, y_max
    if mode == 'tlwh':
        return x_min, y_min, x_max - x_min, y_max - y_min


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
    generate_coco_annotation(args.personfile, args.transferred)
