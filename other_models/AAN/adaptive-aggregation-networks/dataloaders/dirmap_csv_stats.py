import sys
import pandas as pd

# Check if the name/path of the file to analyze was given as a command line argument
if len(sys.argv) > 1:
    DIRMAP_FILE = sys.argv[1]
else:
    DIRMAP_FILE = "ilab2mlight_dirmap.csv"

print("Showing stats for file " + DIRMAP_FILE + ":")

img_df = pd.read_csv(DIRMAP_FILE)

print("Top few rows of dirmap.csv:")
print(img_df.head())

print("Number of examples per class:")
print(img_df.groupby(["class"])["im_num"].count())
print("-----------")

print("Avg number of images per session: {}".format(img_df.groupby(["class", "object", "session"])["im_num"].count().mean()))
print("Min number of images per session: {}".format(img_df.groupby(["class", "object", "session"])["im_num"].count().min()))
print("Max number of images per session: {}".format(img_df.groupby(["class", "object", "session"])["im_num"].count().max()))
print("-----------")
print("Avg number of sessions per object: {}".format(img_df.groupby(["class", "object"])["session"].nunique().mean()))
print("Min number of sessions per object: {}".format(img_df.groupby(["class", "object"])["session"].nunique().min()))
print("Max number of sessions per object: {}".format(img_df.groupby(["class", "object"])["session"].nunique().max()))
print("-----------")
img_df["session_all"] = img_df.groupby(["class", "object", "session"]).ngroup()
print("Avg number of sessions per class: {}".format(img_df.groupby(["class"])["session_all"].nunique().mean()))
print("Min number of sessions per class: {}".format(img_df.groupby(["class"])["session_all"].nunique().min()))
print("Max number of sessions per class: {}".format(img_df.groupby(["class"])["session_all"].nunique().max()))
print("-----------")

# if "ilab2mlight" in DIRMAP_FILE:
#     for prop in ["class", "object", "background", "camera", "rotation", "lighting", "im_num"]:
#         print(str(img_df[prop].nunique()) + " unique " + prop + " values")
#         unique = list(img_df[prop].unique())
#         unique.sort()
#         print(unique)
#         print("-----")

print("Number of objects/instances by class:")
print(img_df.groupby(["class"])["object"].nunique())
print("---------------")

img_df["session_all"] = img_df.groupby(["class", "object", "session"]).ngroup()
print("Session count: {}".format(img_df["session_all"].nunique()))
print("AFTER GETTING RID OF SESSIONS WITH <15 IMAGES (applicable for toybox and ilab2mlight):")
img_df["ims_per_sess"] = img_df["im_num"].groupby([img_df["class"], img_df["object"], img_df["session"]]).transform("max")
img_df_trimmed = img_df.groupby("session_all").filter(lambda sess: sess["ims_per_sess"].min() >= 15)
img_df_trimmed.reset_index(drop=True, inplace=True) # reset index

img_df_trimmed["session_all"] = img_df_trimmed.groupby(["class", "object", "session"]).ngroup()
print("Session count: {}".format(img_df_trimmed["session_all"].nunique()))
