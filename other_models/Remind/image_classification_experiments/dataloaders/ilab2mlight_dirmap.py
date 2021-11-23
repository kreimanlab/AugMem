import os
import sys
import pandas as pd

import ilab2mlight_helper

# USAGE: python ilab2mlight_dirmap.py <path to iLab-2M-Light dataset directory>
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "./../data/toybox"

data_subdirs = ["train_img"]

ilab2mlight_class_inds = pd.read_csv(
    'ilab2mlight_classes.csv', index_col=0, usecols=["class", "index"], squeeze=True).to_dict()


imgs_list = []
for subdir in data_subdirs:
    img_files = [f for f in os.listdir(os.path.join(DATA_DIR, subdir)) if f.endswith(".jpg")]
    for fname in img_files:
        imgs_list.append({
            "class": fname.split("-")[0],
            "instance": int(fname.split("-i")[1].split("-")[0]),
            "background": int(fname.split("-b")[1].split("-")[0]),
            "camera": int(fname.split("-c")[1].split("-")[0]),
            "rotation": int(fname.split("-r")[1].split("-")[0]),
            "lighting": int(fname.split("-l")[1].split("-")[0]),
            "im_path": os.path.join(subdir, fname)
        })

img_df = pd.DataFrame(imgs_list)

# Sort the dirmap and assign contiguous instance, background, image indices
img_df = ilab2mlight_helper.idx_sort_ilab2mlight_dirmap(img_df)

img_df.to_csv("ilab2mlight_dirmap_all.csv")
print(img_df.head())

