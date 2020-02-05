import os, sys
import re
import time
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import cv2
from tqdm import tqdm

# PLEASE CHANGE OPTIONS BEFORE YOU START!
options = {
    'output_snippet_dir': "/projectnb/saenkog/awong1/dataset/kitti/processed",
    "input_kitti_img_path": "/projectnb/saenkog/awong1/dataset/kitti/training/image_02",
    "input_kitti_label_path": "/projectnb/saenkog/awong1/dataset/kitti/training/label_02",
}

# Load label txt for video
HEADER = [
    "frame",
    "track_id",
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_l",
    "bbox_t",
    "bbox_r",
    "bbox_b",
    "dim_h",
    "dim_w",
    "dim_l",
    "loc_x",
    "loc_y",
    "loc_z",
    "rot_y",
    "score"
]

lab_dir = options["input_kitti_label_path"]
label = pd.read_csv(lab_dir + "/0000.txt",  sep=" ", names=HEADER)

# Traverse training set and label

train_dir = options["input_kitti_img_path"]

# image_class_info = {}
image_class_info = pd.DataFrame(columns=["video_id", "frame_id", "class", "image_path"])

count = 0
for dirname in sorted(os.listdir(train_dir)):
    tot_frame = 0
    # Get corresponding train labels
    label = pd.read_csv(options["input_kitti_label_path"] + "/" + dirname + ".txt", sep=" ", names=HEADER)
    count += 1

    # Traverse image dir
    for filename in sorted(os.listdir(train_dir + "/" + dirname)):
        tot_frame += 1
        image_path = train_dir + "/" + dirname + "/" + filename

        # parse frame id from filename
        cur_frame_id = int(re.sub(r"\.png", "", filename))
        
        # select label rows with corresponding frame id
        cur_label = label[label["frame"] == cur_frame_id]
        types = list(cur_label["type"])
        
        pedestrian_labels = cur_label[cur_label["type"] == "Pedestrian"]
        bbox_l = ",".join([str(v) for v in pedestrian_labels["bbox_l"].tolist()])
        bbox_t = ",".join([str(v) for v in pedestrian_labels["bbox_t"].tolist()])
        bbox_r = ",".join([str(v) for v in pedestrian_labels["bbox_r"].tolist()])
        bbox_b = ",".join([str(v) for v in pedestrian_labels["bbox_b"].tolist()])
        
        new_row = {"video_id": dirname,
                   "frame_id": cur_frame_id,
                   "class": 0,
                   "image_path": image_path,
                   "bbox_l": bbox_l,
                   "bbox_t": bbox_t,
                   "bbox_r": bbox_r,
                   "bbox_b": bbox_b}
        
        # detect class-0 / class-1(w/t pedestrian)
        if "Pedestrian" in types:
            new_row["class"] = 1
        
        image_class_info = image_class_info.append(new_row, ignore_index=True)
        
        # print("  %s => types: [%s]" % (filename, ", ".join(types)))
    print("Total frames for video %s: %i" % (dirname, tot_frame))


# Class 1 & class 0 count
counts = image_class_info.groupby(["class"]).size()
ratio = counts[1]/counts[0]

# Balance class 0 and 1
a = image_class_info[image_class_info["class"] == 0].sample(frac=ratio)
b = image_class_info[image_class_info["class"] == 1].sample(frac=1)
balanced_image_class_info = a.append(b)
balanced_image_class_info

# Get the average size of the image
accum_shape = None
for path in tqdm(balanced_image_class_info["image_path"].tolist()[:]):
    img = mpimg.imread(path)
    if accum_shape is None:
        accum_shape = np.array(img.shape)
    else:
        accum_shape += np.array(img.shape)
accum_shape = accum_shape / len(balanced_image_class_info["image_path"].tolist())
accum_shape

# Get the average size of the image
accum_shape = None
for path in tqdm(balanced_image_class_info["image_path"].tolist()[:]):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if accum_shape is None:
        accum_shape = np.array(img.shape)
    else:
        accum_shape += np.array(img.shape)
accum_shape = accum_shape / len(balanced_image_class_info["image_path"].tolist())
accum_shape

# Split into Train/Val/Test
# TODO: Cross validation?

shuffled_image_class_info = balanced_image_class_info.sample(frac = 1.0)

TRAIN_SIZE = int(len(shuffled_image_class_info) * 0.7)
VAL_SIZE = int(len(shuffled_image_class_info) * 0.1)
TEST_SIZE = int(len(shuffled_image_class_info) * 0.2)


train_img_class_info = shuffled_image_class_info[:TRAIN_SIZE]
val_img_class_info = shuffled_image_class_info[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_img_class_info = shuffled_image_class_info[TRAIN_SIZE+VAL_SIZE:]

print("Train size: %s" % len(train_img_class_info))
print("Val size: %s" % len(val_img_class_info))
print("Test size: %s" % len(test_img_class_info))

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
#     assert bb1['x1'] < bb1['x2']
#     assert bb1['y1'] < bb1['y2']
#     assert bb2['x1'] < bb2['x2']
#     assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    #iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    #assert iou >= 0.0
    #assert iou <= 1.0
    if float(bb2_area) == 0.0:
        return 0.0
    return float(intersection_area)/float(bb2_area)

a1 = {"x1": 400, "x2": 624, "y1": 0, "y2": 224}
a2 = {"x1": 230, "x2": 261, "y1": 169, "y2": 255} #514 573 88 195 #230 261 169 255
get_iou(a1, a2)

# from bbox import BBox2D, XYXY
# from bbox.metrics import jaccard_index_2d

# a1 = BBox2D([400, 0, 624, 224], mode=XYXY)
# a2 = BBox2D([230, 169, 230, 255], mode=XYXY)
# jaccard_index_2d(a1, a2)


cropped_imgs = []
has_pedestrian = []

disp_img = True

# for i, (index, row) in tqdm(enumerate(balanced_image_class_info[balanced_image_class_info["class"] == 1].iterrows())):
for i, (index, row) in tqdm(enumerate(train_img_class_info.iterrows())):
    img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_resize = cv2.resize(img, (int(img.shape[1]*224/img.shape[0]), 224))
    if disp_img:
        plt.imshow(img_resize)    
    # mask = np.zeros((img_resize.shape[0], img_resize.shape[1]))
    
    # Cropping and detecting bounding box IoU
    crop_idx = np.random.randint(0, img_resize.shape[1]-224)
    img_crop = img_resize[:, crop_idx:crop_idx+224]
    has_full_pedestrian = False

    # Using bounding box info
    if row["bbox_l"] != "":
        ax = plt.gca()
        for tup in zip(row["bbox_l"].split(","), row["bbox_t"].split(","), row["bbox_r"].split(","), row["bbox_b"].split(",")):
            bbox_l = int(float(tup[0])*740/1240)
            bbox_t = int(float(tup[1])*224/375)
            bbox_r = int(float(tup[2])*740/1240)
            bbox_b = int(float(tup[3])*224/375)

            # mask[bbox_t: bbox_b, bbox_l:bbox_r] = 1.0

            # Calculate IoU?
            crop_box = {"x1": crop_idx, "x2": crop_idx+224, "y1": 0, "y2": 224}
            obj_box = {"x1": bbox_l, "x2": bbox_r, "y1": bbox_t, "y2": bbox_b}
            coverage = get_iou(crop_box, obj_box)
            if coverage > 0.79:
                has_full_pedestrian = True
            rect = patches.Rectangle((bbox_l, bbox_t), (bbox_r - bbox_l), (bbox_b - bbox_t), linewidth=2, edgecolor='lawngreen', facecolor='none', label="%.2f"%coverage)
            ax.add_patch(rect)
    if disp_img:
        plt.show()    
        plt.imshow(img_crop)
        print(has_full_pedestrian)
        plt.show()
    
    cropped_imgs.append(img_crop)
    has_pedestrian.append(has_full_pedestrian)
    
#     if i > 100: break

has_pedestrian = np.array(has_pedestrian)
print("no_pedestrian count: ", np.sum(has_pedestrian == False))
print("has_pedestrian count: ", np.sum(has_pedestrian == True))

has_pedestrian_idxs = []
no_pedestrian_idxs = []

import cv2

for i in range(len(has_pedestrian)):
    if has_pedestrian[i] == True:
        has_pedestrian_idxs.append(i)
        cv2.imwrite(options["output_snippet_dir"] + "/has_pedestrian/" + str(i) + ".png", cropped_imgs[i])
    else:
        no_pedestrian_idxs.append(i)
        cv2.imwrite(options["output_snippet_dir"] + "/no_pedestrian/" + str(i) + ".png", cropped_imgs[i])

print("has_pedestrian_idxs:", has_pedestrian_idxs)
print("no_pedestrian_idxs:", no_pedestrian_idxs)

for i in range(25):
    if has_pedestrian[i] == False:
        continue
    print(has_pedestrian[i])
    plt.imshow(cropped_imgs[i])
    plt.show()
