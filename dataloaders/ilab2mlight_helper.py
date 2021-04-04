def idx_sort_ilab2mlight_dirmap(img_df):
    img_df = img_df.sort_values(by=["class", "instance", "background", "camera", "rotation", "lighting"], ignore_index=True)

    img_df["instance_idx"] = img_df.groupby(["class", "instance"]).ngroup()
    img_df["instance_idx_min"] = img_df["instance_idx"].groupby(img_df["class"]).transform("min")
    img_df["object"] = img_df["instance_idx"] - img_df["instance_idx_min"] + 1

    img_df["session_idx"] = img_df.groupby(["class", "instance", "background"]).ngroup()
    img_df["session_idx_min"] = img_df["session_idx"].groupby([img_df["class"], img_df["instance"]]).transform("min")
    img_df["session"] = img_df["session_idx"] - img_df["session_idx_min"] + 1

    img_df["im_num"] = img_df.groupby(["class", "object", "session"]).cumcount() + 1

    img_df = img_df[["class", "object", "session", "im_num", "im_path", "instance", "background", "camera", "rotation", "lighting"]]

    img_df.reset_index(drop=True, inplace=True)  # reset index

    return img_df
