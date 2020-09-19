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
    parser.add_argument("json", help="Which JSON file should be used?", default="person_bbox_train.json")
    parser.add_argument("imagedir", help="Which image directory should be used?", default="image_train")

    args = parser.parse_args()

    image_root = args.directory
    person_anno_file = args.json
    annomode = 'person'

    outpath = 'split'
    outannofile = 'split.json'
    split = ImgSplit(image_root, person_anno_file, annomode, outpath, outannofile, args.imagedir)
    split.splitdata(0.5)