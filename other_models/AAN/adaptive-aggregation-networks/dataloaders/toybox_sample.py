import pandas as pd
import numpy as np

# PLEASE NOTE: this takes in toybox_dirmap_unbalanced.csv, which should already have toybox sampled at 1 fps.

img_df = pd.read_csv("toybox_dirmap_unbalanced.csv")

OBJ_PER_CLASS = 30
SESS_PER_OBJ = 10
IMS_PER_SESS = 15

print("Filtering to keep only {} images per session...".format(IMS_PER_SESS))
# Get only the first IMS_PER_SESS images for each session
img_df = img_df[img_df.im_num.isin(np.arange(1, IMS_PER_SESS+1, 1))]


print("Augmenting short sessions...")
# For each session with fewer than the required number of frames, duplicate the existing frames and run them in reverse,
# then add them to the end of the session, only using the final frame once. For example, if a session has 9 frames and
# you need 15, it will give you the images 1,2,3,4,5,6,7,8,9,8,7,6,5,4,3

# Get all sessions that have fewer than IMS_PER_SESS images
short_sessions_only = img_df.groupby(["class", "object", "session"]).filter(lambda sess: sess["im_num"].max() < IMS_PER_SESS)

gb = short_sessions_only.groupby(["class", "object", "session"])
short_sessions = [gb.get_group(x) for x in gb.groups]

pd.set_option('display.max_rows', 100)

for sess in short_sessions:
    print("ORIGINAL SHORT SESSION:")
    print(sess)
    print("--------------")
    reversed_sess = sess.iloc[::-1]
    reversed_sess = reversed_sess[reversed_sess["im_num"] != len(reversed_sess)]
    aug_sess = pd.concat([sess, reversed_sess])
    print("AUGMENTED SESSION (im_num will be reindexed, and truncated to a maximum of {}):".format(IMS_PER_SESS))
    print(aug_sess)
    print("--------------")
    aug_sess.reset_index(drop=True, inplace=True)
    aug_sess["im_num"] = aug_sess.index + 1
    aug_sess = aug_sess[aug_sess["im_num"] <= IMS_PER_SESS]
    print("FINAL AUGMENTED SESSION:")
    print(aug_sess)
    print("--------------")

    img_df = pd.concat([img_df, aug_sess])

img_df = img_df.drop_duplicates()
img_df = img_df.sort_values(by=["class", "object", "session", "im_num"], ignore_index=True)
img_df.reset_index(drop=True, inplace=True)

print("Removing objects that have fewer than 10 sessions")
img_df = img_df.groupby(["class", "object"]).filter(lambda obj: obj["session"].nunique() >= SESS_PER_OBJ)

print("Removing the final object from every class, except for classes that already have fewer than 30 objects")
img_df["obj_per_class"] = img_df.groupby(["class"])["object"].transform("nunique")
img_df = img_df.groupby(["class", "object"]).filter(
    lambda obj: obj["obj_per_class"].min() < OBJ_PER_CLASS or obj["object"].min() != OBJ_PER_CLASS)
# Reindex objects
img_df["object_idx"] = img_df.groupby(["class", "object"]).ngroup()
img_df["object_idx_min"] = img_df.groupby(["class"])["object_idx"].transform("min")
img_df["object"] = img_df["object_idx"] - img_df["object_idx_min"] + 1
img_df = img_df[["class", "object", "session", "im_num", "im_path"]]

print("Number of rows in final dirmap: {}".format(len(img_df)))

img_df.to_csv("toybox_dirmap.csv")
