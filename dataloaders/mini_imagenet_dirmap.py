import os
import sys
import pandas as pd


# USAGE: python mini_imagenet_dirmap.py <path to mini_imagenet dataset directory>
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "./../data/mini_imagenet"

orig_csvs = {
    "train": pd.read_csv(os.path.join(DATA_DIR, "train.csv")),
    "valid": pd.read_csv(os.path.join(DATA_DIR, "val.csv")),
    "test": pd.read_csv(os.path.join(DATA_DIR, "test.csv")),
}

imgs_list = []
for test_train_val_idx, (key, df) in enumerate(orig_csvs.items()):
    for row_idx, row in df.iterrows():
        out_dir = os.path.join(DATA_DIR, "preprocessed", row["label"])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        os.system("cp " + os.path.join(DATA_DIR, "images", row["filename"]) + " " + os.path.join(out_dir, row["filename"]))
        imgs_list.append({
            "class": row["label"],
            "object": 0,
            "session": test_train_val_idx,
            "im_path": os.path.join("preprocessed", row["label"], row["filename"])
        })

img_df = pd.DataFrame(imgs_list)
img_df = img_df.sort_values(by=["class", "object", "session"], ignore_index=True)
img_df["im_num"] = img_df.groupby(["class", "object", "session"]).cumcount() + 1

img_df.to_csv("mini_imagenet_dirmap.csv")
print(img_df.head())

