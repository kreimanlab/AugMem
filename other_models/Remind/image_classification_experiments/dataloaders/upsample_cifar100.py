import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm


# USAGE: python resize_cifar100.py <path to cifar100 dataset directory> <path to output (resized cifar100)>
# Organized cifar100 directory can be created using cifar2png: https://github.com/knjcode/cifar2png
# cifar100_dirmap.csv (required for this script) can then be created by running cifar100_dirmap.py
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "./../data/cifar100"

if len(sys.argv) > 2:
    OUT_DIR = sys.argv[2]
else:
    OUT_DIR = "./../data/cifar100_resized"

try:
    img_df = pd.read_csv("cifar100_dirmap.csv")
except FileNotFoundError:
    print(
        "ERROR: cifar100_dirmap.csv (required for this script) was not found. It can be created by "
        + "running cifar100_dirmap.py after retrieving cifar100 png images using https://github.com/knjcode/cifar2png"
    )
    exit()

for _, row in tqdm(img_df.iterrows()):
    in_path = os.path.join(DATA_DIR, row['im_path'])

    im = Image.open(in_path)
    resized_im = im.resize((224, 224), resample=Image.BILINEAR)

    out_path = os.path.join(OUT_DIR, row['im_path'])
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    resized_im.save(out_path)






