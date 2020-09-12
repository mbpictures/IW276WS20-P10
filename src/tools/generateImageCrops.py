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
    parser.add_argument("directory", required=True, help="Directory in which your dataset is stored (this folder contains an folder called 'image_annos' and 'image_train'")

    args = parser.parse_args()

    image_root = args.directory
    person_anno_file = 'person_bbox_train.json'
    annomode = 'person'

    outpath = 'split'
    outannofile = 'split.json'
    split = ImgSplit(image_root, person_anno_file, annomode, outpath, outannofile)
    split.splitdata(0.5)