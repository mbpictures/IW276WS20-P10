# --------------------------------------------------------
# Image and annotations splitting modules for PANDA
# Written by Wang Xueyang  (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import os
import cv2
import json
import copy
from collections import defaultdict


class ImgSplit():
    def __init__(self,
                 base_path,
                 annotations_file,
                 annotation_mode,
                 out_path,
                 out_annotations_file,
                 code='utf-8',
                 gap=100,
                 sub_width=1000,
                 sub_height=500,
                 thresh=0.7,
                 out_extension='.jpg'
                 ):
        """
        :param base_path: base directory for panda image data and annotations
        :param annotations_file: annotation file path
        :param annotation_mode:the type of annotation, which can be 'person', 'vehicle', 'headbbox' or 'headpoint'
        :param out_path: output base path for panda data
        :param out_annotations_file: output file path for annotation
        :param code: encoding format of txt file
        :param gap: overlap between two patches
        :param sub_width: sub-width of patch
        :param sub_height: sub-height of patch
        :param thresh: the square thresh determine whether to keep the instance which is cut in the process of split
        :param out_extension: ext for the output image format
        """
        self.base_path = base_path
        self.annotations_file = annotations_file
        self.annotation_mode = annotation_mode
        self.out_path = out_path
        self.out_annotations_file = out_annotations_file
        self.code = code
        self.gap = gap
        self.sub_width = sub_width
        self.sub_height = sub_height
        self.slide_width = self.sub_width - self.gap
        self.slide_height = self.sub_height - self.gap
        self.thresh = thresh
        self.image_path = os.path.join(self.base_path, 'image_train')
        self.annotations_path = os.path.join(self.base_path, 'image_annos', annotations_file)
        self.out_image_path = os.path.join(self.out_path, 'image_train')
        self.out_annotations_path = os.path.join(self.out_path, 'image_annos')
        self.out_extension = out_extension
        if not os.path.exists(self.out_image_path):
            os.makedirs(self.out_image_path)
        if not os.path.exists(self.out_annotations_path):
            os.makedirs(self.out_annotations_path)
        self.annotations = defaultdict(list)
        self.load_annotations()

    def load_annotations(self):
        print('Loading annotation json file: {}'.format(self.annotations_path))
        with open(self.annotations_path, 'r') as load_f:
            annodict = json.load(load_f)
        self.annotations = annodict

    def split_data(self, scale, image_request=None, image_filters=[]):
        """
        :param scale: resize rate before cut
        :param image_request: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param image_filters: essential keywords in image name
        """
        if image_request is None or not isinstance(image_request, list):
            image_names = list(self.annotations.keys())
        else:
            image_names = image_request

        split_annotations = {}
        for image_name in image_names:
            keep = False
            for image_filter in image_filters:
                if image_filter in image_name:
                    keep = True
            if image_filters and not keep:
                continue
            split_dict = self.split_single(image_name, scale)
            split_annotations.update(split_dict)

        # add image id
        image_id = 1
        for image_name in split_annotations.keys():
            split_annotations[image_name]['image id'] = image_id
            image_id += 1
        # save new annotation for split images
        out_dir = os.path.join(self.out_annotations_path, self.out_annotations_file)
        with open(out_dir, 'w', encoding=self.code) as f:
            dict_str = json.dumps(split_annotations, indent=2)
            f.write(dict_str)

    @staticmethod
    def load_image(image_path):
        """
        :param image_path: the path of image to load
        :return: loaded img object
        """
        print('filename:', image_path)
        if not os.path.exists(image_path):
            print('Can not find {}, please check local dataset!'.format(image_path))
            return None
        img = cv2.imread(image_path)
        return img

    def split_single(self, image_name, scale):
        """
        split a single image and ground truth
        :param image_name: image name
        :param scale: the resize scale for the image
        :return:
        """
        image_path = os.path.join(self.image_path, image_name)
        img = self.load_image(image_path)
        if img is None:
            return
        image_dict = self.annotations[image_name]
        object_list = image_dict['objects list']

        # re-scale image if scale != 1
        if scale != 1:
            resize_image = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resize_image = img
        image_height, image_width = resize_image.shape[:2]

        # split image and annotation in sliding window manner
        out_base_name = image_name.replace('/', '_').replace(' ', '_').split('.')[0] + '___' + str(scale) + '__'
        sub_image_annotations = {}
        left, up = 0, 0
        while left < image_width:
            if left + self.sub_width >= image_width:
                left = max(image_width - self.sub_width, 0)
            up = 0
            while up < image_height:
                if up + self.sub_height >= image_height:
                    up = max(image_height - self.sub_height, 0)
                right = min(left + self.sub_width, image_width - 1)
                down = min(up + self.sub_height, image_height - 1)
                coordinates = left, up, right, down
                sub_image_name = out_base_name + str(left) + '__' + str(up) + self.out_extension
                # split annotations according to annotation mode
                if self.annotation_mode == 'person':
                    new_objects_list = self.person_annotations_split(object_list, image_width, image_height, coordinates)
                elif self.annotation_mode == 'vehicle':
                    new_objects_list = self.vehicle_annotations_split(object_list, image_width, image_height, coordinates)
                elif self.annotation_mode == 'headbbox':
                    new_objects_list = self.head_bbox_annotations_split(object_list, image_width, image_height, coordinates)
                elif self.annotation_mode == 'headpoint':
                    new_objects_list = self.head_point_annotations_split(object_list, image_width, image_height, coordinates)

                if up + self.sub_height >= image_height:
                    break
                else:
                    up = up + self.slide_height
                
                # no persons on image? SKIP!
                if len(new_objects_list) == 0:
                    continue

                self.save_sub_image(resize_image, sub_image_name, coordinates)
                sub_image_annotations[sub_image_name] = {
                    "image size": {
                        "height": down - up + 1,
                        "width": right - left + 1
                    },
                    "objects list": new_objects_list
                }
                
            if left + self.sub_width >= image_width:
                break
            else:
                left = left + self.slide_width

        return sub_image_annotations

    def judge_rect(self, rect_dict, image_width, image_height, coordinates):
        left, up, right, down = coordinates
        x_min = int(rect_dict['tl']['x'] * image_width)
        y_min = int(rect_dict['tl']['y'] * image_height)
        x_max = int(rect_dict['br']['x'] * image_width)
        y_max = int(rect_dict['br']['y'] * image_height)
        square = (x_max - x_min) * (y_max - y_min)

        if (x_max <= left or right <= x_min) and (y_max <= up or down <= y_min):
            intersection = 0
        else:
            lens = min(x_max, right) - max(x_min, left)
            wide = min(y_max, down) - max(y_min, up)
            intersection = lens * wide

        return intersection and intersection / (square + 1e-5) > self.thresh

    @staticmethod
    def restrain_rect(rect_dict, image_width, image_height, coordinates):
        left, up, right, down = coordinates
        x_min = int(rect_dict['tl']['x'] * image_width)
        y_min = int(rect_dict['tl']['y'] * image_height)
        x_max = int(rect_dict['br']['x'] * image_width)
        y_max = int(rect_dict['br']['y'] * image_height)
        x_min = max(x_min, left)
        x_max = min(x_max, right)
        y_min = max(y_min, up)
        y_max = min(y_max, down)
        return {
            'tl': {
                'x': (x_min - left) / (right - left),
                'y': (y_min - up) / (down - up)
            },
            'br': {
                'x': (x_max - left) / (right - left),
                'y': (y_max - up) / (down - up)
            }
        }

    @staticmethod
    def judge_point(rect_dict, image_width, image_height, coordinates):
        left, up, right, down = coordinates
        x = int(rect_dict['x'] * image_width)
        y = int(rect_dict['y'] * image_height)

        if left < x < right and up < y < down:
            return True
        else:
            return False

    @staticmethod
    def restrain_point(rect_dict, image_width, image_height, coordinates):
        left, up, right, down = coordinates
        x = int(rect_dict['x'] * image_width)
        y = int(rect_dict['y'] * image_height)
        return {
            'x': (x - left) / (right - left),
            'y': (y - up) / (down - up)
        }

    def person_annotations_split(self, objects_list, image_width, image_height, coordinates):
        new_objects_list = []
        for object_dict in objects_list:
            object_category = object_dict['category']
            if object_category == 'person':
                pose = object_dict['pose']
                riding = object_dict['riding type']
                age = object_dict['age']
                full_rect = object_dict['rects']['full body']
                visible_rect = object_dict['rects']['visible body']
                head_rect = object_dict['rects']['head']
                # only keep a person whose 3 box all satisfy the requirement
                if self.judge_rect(full_rect, image_width, image_height, coordinates) & \
                   self.judge_rect(visible_rect, image_width, image_height, coordinates) & \
                   self.judge_rect(head_rect, image_width, image_height, coordinates):
                    new_objects_list.append({
                        "category": object_category,
                        "pose": pose,
                        "riding type": riding,
                        "age": age,
                        "rects": {
                            "head": self.restrain_rect(head_rect, image_width, image_height, coordinates),
                            "visible body": self.restrain_rect(visible_rect, image_width, image_height, coordinates),
                            "full body": self.restrain_rect(full_rect, image_width, image_height, coordinates)
                        }
                    })
            else:
                rect = object_dict['rect']
                if self.judge_rect(rect, image_width, image_height, coordinates):
                    new_objects_list.append({
                        "category": object_category,
                        "rect": self.restrain_rect(rect, image_width, image_height, coordinates)
                    })
        return new_objects_list

    def vehicle_annotations_split(self, objects_list, image_width, image_height, coordinates):
        new_objects_list = []
        for object_dict in objects_list:
            object_category = object_dict['category']
            rect = object_dict['rect']
            if self.judge_rect(rect, image_width, image_height, coordinates):
                new_objects_list.append({
                    "category": object_category,
                    "rect": self.restrain_rect(rect, image_width, image_height, coordinates)
                })
        return new_objects_list

    def head_bbox_annotations_split(self, objects_list, image_width, image_height, coordinates):
        new_objects_list = []
        for object_dict in objects_list:
            rect = object_dict['rect']
            if self.judge_rect(rect, image_width, image_height, coordinates):
                new_objects_list.append({
                    "rect": self.restrain_rect(rect, image_width, image_height, coordinates)
                })
        return new_objects_list

    def head_point_annotations_split(self, object_list, image_width, image_height, coordinates):
        new_object_list = []
        for object_dict in object_list:
            rect = object_dict['rect']
            if self.judge_point(rect, image_width, image_height, coordinates):
                new_object_list.append({
                    "rect": self.restrain_point(rect, image_width, image_height, coordinates)
                })
        return new_object_list

    def save_sub_image(self, img, sub_image_name, coordinates):
        left, up, right, down = coordinates
        sub_image = copy.deepcopy(img[up: down, left: right])
        out_dir = os.path.join(self.out_image_path, sub_image_name)
        cv2.imwrite(out_dir, sub_image)