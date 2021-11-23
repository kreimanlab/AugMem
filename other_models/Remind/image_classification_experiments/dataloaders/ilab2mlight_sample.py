import pandas as pd
import random

import ilab2mlight_helper

img_df = pd.read_csv("ilab2mlight_dirmap_all.csv")

print("SUBSAMPLING THE ILAB2MLIGHT DATASET:")

random.seed(0) # Random seed for reproducibility
FINAL_OBJECTS_PER_CLASS = 28
FINAL_SESSIONS_PER_OBJECT = 8
FINAL_IMGS_PER_SESSION = 15

# Remove "semi" class, which has few objects that have more than 8 sessions
sampled = img_df[img_df["class"] != "semi"]

# Re-index dataset
sampled = ilab2mlight_helper.idx_sort_ilab2mlight_dirmap(sampled)

# Remove all sessions with fewer than FINAL_IMAGES_PER_SESSION images
sampled = sampled.groupby(["class", "object", "session"]).filter(lambda sess: sess["im_num"].max() >= FINAL_IMGS_PER_SESSION)

# Re-index dataset
sampled = ilab2mlight_helper.idx_sort_ilab2mlight_dirmap(sampled)

# Keep only objects with more than n sessions per object
sampled = sampled.groupby(["class", "object"]).filter(lambda obj: obj["session"].max() >= FINAL_SESSIONS_PER_OBJECT)

# Re-index dataset
sampled = ilab2mlight_helper.idx_sort_ilab2mlight_dirmap(sampled)

# Keep only the first n sessions per object (to make objects have a consistent number of sessions)
sampled = sampled[sampled["session"] <= FINAL_SESSIONS_PER_OBJECT]

# Keep only the first n objects in each class
sampled = sampled[sampled["object"] <= FINAL_OBJECTS_PER_CLASS]

# From each session, keep only 15 images in a randomly chosen contiguous sequence
sampled["sess_start_num"] = sampled.groupby(["class", "object", "session"])["im_num"].transform(lambda sess: random.randint(1, sess.max()-FINAL_IMGS_PER_SESSION+1))
sampled = sampled.loc[(sampled["im_num"] >= sampled["sess_start_num"]) & (sampled["im_num"] < sampled["sess_start_num"] + FINAL_IMGS_PER_SESSION)]

# Re-index dataset
sampled = ilab2mlight_helper.idx_sort_ilab2mlight_dirmap(sampled)

print("Final shape of sampled dirmap:")
print(sampled.shape)

sampled.to_csv("ilab2mlight_dirmap_massed.csv")