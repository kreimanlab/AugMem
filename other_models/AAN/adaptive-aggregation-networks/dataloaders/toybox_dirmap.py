import os
import sys
import pandas as pd


# USAGE: python toybox_dirmap.py <path to toybox dataset directory>
# Toybox dataset directory should contain directories "household", "animals", and "vehicles"
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "./../data/toybox"

# If true, generate the images from video files. if False, just make the toybox_dirmap.csv
MAKE_IMAGES = True

# If False, sample toybox images at 1fps, instead of getting every single image (which is what you get with "True" here)
ALL_FRAMES = False

# In toybox dataset, there are 3 supercategories, each containing 4 categories
toybox_cats = {
    "households": ["cup", "mug", "spoon", "ball"],
    "animals": ["duck", "cat", "horse", "giraffe"],
    "vehicles": ["car", "truck", "airplane", "helicopter"],
}

NUM_OBJS_PER_CAT = 30

toybox_class_inds = pd.read_csv('toybox_classes.csv', index_col=0, usecols=["class", "index"], squeeze=True).to_dict()
toybox_session_inds = pd.read_csv('toybox_sessions.csv', index_col=0, squeeze=True).to_dict()


def make_imgs(vid_path, img_out_folder, name_stub, all_frames=ALL_FRAMES, fps=1):
    if all_frames:
        command = "ffmpeg -i " + vid_path + " -s 224x224 " + img_out_folder + "/" + name_stub + "%d.png"
    else:
        command = "ffmpeg -i " + vid_path + " -s 224x224 -vf fps=" + str(fps) + " " + img_out_folder + "/" + name_stub + "%d.png"
    os.system(command)


imgs_list = []
for supercat in toybox_cats.keys():
    for category in toybox_cats[supercat]:
        for i in range(1, NUM_OBJS_PER_CAT+1):
            obj_dir_end = os.path.join(supercat, (category + "_" + "{0:0=2d}".format(i) + "_pivothead"))

            output_dir = os.path.join(DATA_DIR, "images", obj_dir_end)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            obj_dir = os.path.join(DATA_DIR, obj_dir_end)

            if MAKE_IMAGES:
                mp4_list = [f for f in os.listdir(obj_dir) if f.endswith(".mp4")]
                for vid in mp4_list:
                    make_imgs(os.path.join(obj_dir, vid), output_dir, os.path.splitext(vid)[0] + "_")

            img_list = [f for f in os.listdir(output_dir)
                        if f.endswith(".png") and
                        os.path.splitext(f)[0].split("_")[-2] in toybox_session_inds.keys()]
            for img in img_list:
                imgs_list.append({
                    "class": toybox_class_inds[category],
                    "object": i,
                    "session": int(toybox_session_inds[os.path.splitext(img)[0].split("_")[-2]]),
                    "im_num": int(os.path.splitext(img)[0].split("_")[-1]),
                    "im_path": os.path.join(output_dir, img).split("/images/")[-1]
                })

print("Image extraction/cataloging steps complete, converting to dataframe... ")
img_df = pd.DataFrame(imgs_list)
img_df = img_df.sort_values(by=["class", "object", "session", "im_num"], ignore_index=True)
if ALL_FRAMES:
    img_df.to_csv("toybox_dirmap_allframes.csv")
else:
    img_df.to_csv("toybox_dirmap_unbalanced.csv")
print(img_df.head())

