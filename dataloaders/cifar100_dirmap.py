import os
import sys
import pandas as pd


# USAGE: python cifar100_dirmap.py <path to cifar100 dataset directory>
# Organized cifar100 directory can be created using cifar2png: https://github.com/knjcode/cifar2png
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "./../data/cifar100"

# Get class names
class_names = [
    file for file in os.listdir(os.path.join(DATA_DIR, "train"))
    if os.path.isdir(os.path.join(DATA_DIR, "train", file))
]
class_names.sort()
class_dicts = [{"class": class_names[i], "label": i} for i in range(len(class_names))]
pd.DataFrame(class_dicts).to_csv("cifar100_classes.csv", index=False)

image_list = []
for train_test_idx, train_test in enumerate(["train", "test"]):
    for img_class in class_names:
        img_files = [f for f in os.listdir(os.path.join(DATA_DIR, train_test, img_class)) if f.endswith(".png")]
        for fname in img_files:
            image_list.append({
                "class": img_class,
                "object": 0,
                "session": train_test_idx,
                "im_path": os.path.join(train_test, img_class, fname),
            })

img_df = pd.DataFrame(image_list)
img_df = img_df.sort_values(by=["class", "object", "session", "im_path"], ignore_index=True)
img_df["im_num"] = img_df.groupby(["class", "object", "session"]).cumcount() + 1

img_df.to_csv("cifar100_dirmap.csv")
print(img_df.head())
