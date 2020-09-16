# --------------------------------------------------------
# Tool kit function demonstration
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

#### MODIFIED VERSION ####

from ImgSplit import ImgSplit
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("How to use this script: generateImageCrops.py ROOT_DIRECTORY\nROOT_DIRECTORY is the directory which contains your image_train and image_annos folders")
    else:
        image_root = sys.argv[1]
        person_anno_file = 'person_bbox_train.json'
        annomode = 'person'

        outpath = 'split'
        outannofile = 'split.json'
        split = ImgSplit(image_root, person_anno_file, annomode, outpath, outannofile)
        split.splitdata(0.5)