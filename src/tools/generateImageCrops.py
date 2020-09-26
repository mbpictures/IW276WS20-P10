# --------------------------------------------------------
# Tool kit function demonstration
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

#### MODIFIED VERSION ####

from ImgSplit import ImgSplit
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop high res dataset images to lower res cropped images and filter only images containing at least one human")
    parser.add_argument("directory", help="Directory in which your dataset is stored (this folder contains an folder called 'image_annos' and 'image_train'")

    args = parser.parse_args()

    image_root = args.directory
    person_annotations_file = 'person_bbox_train.json'
    annotation_mode = 'person'

    out_path = 'split'
    out_annotations_file = 'split.json'
    split = ImgSplit(image_root, person_annotations_file, annotation_mode, out_path, out_annotations_file)
    split.split_data(0.5)