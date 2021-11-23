import os
import sys
import pandas as pd

# USAGE: python ilab2mlight_distribute_img_dirs.py <path to iLab-2M-Light dataset directory> <path to place distributed version of dataset>
if len(sys.argv) > 2:
    DATA_SOURCE_DIR = sys.argv[1]
    OUT_DIR = sys.argv[2]
else:
    raise ValueError("Please pass a data source directory and an output directory as command line arguments (see README). USAGE: python ilab2mlight_distribute_img_dirs.py <path to iLab-2M-Light dataset directory> <new path to distributed version of dataset>")

IMS_PER_SESS = 15

img_df = pd.read_csv("ilab2mlight_dirmap_massed.csv")

# Make the directory structure
classes = list(img_df["class"].unique())
objects = list(img_df["object"].unique())
sessions = list(img_df["session"].unique())

for cl in classes:
    print("Class: " + cl)
    for obj in objects:
        print("Object: " + str(obj))
        for sess in sessions:
            new_dir_rel = os.path.join(cl, "obj_" + str(obj), "sess_" + str(sess))
            new_dir = os.path.join(OUT_DIR, new_dir_rel)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            sess_mask = (img_df["class"] == cl) & (img_df["object"] == obj) & (img_df["session"] == sess)
            assert len(img_df[sess_mask]) == IMS_PER_SESS, "Unmatched number of images per session: {}".format(len(img_df[sess_mask]))

            for _, row in img_df[sess_mask].iterrows():
                img_name = row["im_path"].split("/")[-1]
                new_img_path = os.path.join(new_dir, img_name)
                os.system("cp " + os.path.join(DATA_SOURCE_DIR, row["im_path"]) + " " + new_img_path)

                new_img_path_rel = os.path.join(new_dir_rel, img_name)
                img_df.loc[sess_mask & (img_df["im_path"] == row["im_path"]), "im_path"] = new_img_path_rel

img_df.to_csv("ilab2mlight_dirmap.csv")