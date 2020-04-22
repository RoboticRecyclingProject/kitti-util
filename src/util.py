import os, sys
import re
import pandas as pd

KITTI_HEADER = [
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

"""
Traverse kitti video frames and labels.
Set image-level label according to pedestrian occurence in scene.
"""
def gen_kitti_frame_bins(image_dir, label_dir):

    print("Processing from %s and %s..." % (image_dir, label_dir))
    # image_class_info = {}
    image_class_info = pd.DataFrame(columns=["video_id", "frame_id", "class", "image_path"])

    for dirname in sorted(os.listdir(image_dir)):
        tot_frame = 0
        # Get corresponding train labels
        label = pd.read_csv(label_dir + dirname + ".txt", sep=" ", names=KITTI_HEADER)
        # Traverse image dir
        for filename in sorted(os.listdir(image_dir + dirname)):
            tot_frame += 1
            image_path = image_dir + dirname + "/" + filename

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
    return image_class_info


def balance_img_class_info(img_class_info):
    counts = img_class_info.groupby(["class"]).size()
    ratio = counts[1]/counts[0]
    print("Class ratio:\n\tclass 0: %i\n\tclass 1: %i" % (counts[0], counts[1]))

    a = img_class_info[img_class_info["class"] == 0].sample(frac=ratio)
    b = img_class_info[img_class_info["class"] == 1].sample(frac=1)
    balanced_image_class_info = a.append(b)
    return balanced_image_class_info

def get_coverage(bb1, bb2):
    """
    Calculate the coverage of two bounding boxes.

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

def collate_fn(seq_list):
    img, target = zip(*seq_list)
    #print("collate", target, len(target))
    #imgT = torch.stack([i for i in img])
    targets = [t for t in target]
    imgT = torch.stack([i for i in img])
#     imgT = [i for i in img]
#     imgT = torch.LongTensor(imgT)
    #print("collate_after", targets)
    return (imgT, targets)

class KittiDataset(Dataset):
    def __init__(self, image_class_info):
        # TODO: maybe put the whole preprocess thing here?
        self.img_class_info = image_class_info
#         self.transform = transforms.Compose([transforms.ToTensor(),
#                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                   std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, index):
        row = self.img_class_info.iloc[index]
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, (740, 224))

        # Cropping to 224x224 considering bounding box labels
        if row["class"] == 1:
            # Parse bboxes in image
            bbox_ls = [int(float(v)*(740/1240)) for v in row["bbox_l"].split(",")]
            bbox_ts = [int(float(v)*(224/375)) for v in row["bbox_t"].split(",")]
            bbox_rs = [int(float(v)*(740/1240)) for v in row["bbox_r"].split(",")]
            bbox_bs = [int(float(v)*(224/375)) for v in row["bbox_b"].split(",")]
            #print("Crop bound: [%.2f ~ %.2f]" % (min(bbox_ls), max(bbox_ls)))
            rand_box_idx = np.random.randint(0, len(bbox_ls))

            l_bnd, r_bnd = max(0, int(bbox_rs[rand_box_idx]-224)), min(img_resize.shape[1]-224, int(bbox_ls[rand_box_idx]))
            if l_bnd > r_bnd:
                l_bnd, r_bnd = r_bnd, l_bnd
            elif l_bnd == r_bnd:
                r_bnd += 1

            #print("Random cropping", (l_bnd, r_bnd))
            crop_idx = np.random.randint(l_bnd, r_bnd)
            #crop_idx = np.random.randint(min(img_resize.shape[1]-224, max(0, int(min(bbox_ls)))), min(int(max(bbox_ls)), img_resize.shape[1]-224)+1)
        else:
            crop_idx = np.random.randint(0, img_resize.shape[1]-224)
        img_crop = img_resize[:, crop_idx:crop_idx+224]
        assert(crop_idx >= 0)


        target = {}
        label = 1.0 if row["class"] == 1 else 0.0
        bboxs = []

        # Using bounding box info
        if row["class"] == 1:
            for (bbox_l, bbox_t, bbox_r, bbox_b) in zip(bbox_ls, bbox_ts, bbox_rs, bbox_bs):
#                 bbox_l = int(bbox_l*(740/1240))
#                 bbox_t = int(bbox_t*(224/375))
#                 bbox_r = int(bbox_r*(740/1240))
#                 bbox_b = int(bbox_b*(224/375))

                # Calculate IoU?
                #crop_box = BBox2D([crop_idx, 0, crop_idx+224, 224], mode=XYXY)
                #obj_box = BBox2D([bbox_l, bbox_t, bbox_r, bbox_b], mode=XYXY)
                crop_box = {"x1": crop_idx, "x2": crop_idx+224, "y1": 0, "y2": 224}
                obj_box = {"x1": bbox_l, "x2": bbox_r, "y1": bbox_t, "y2": bbox_b}
                # Positive tile if any of the pedestrian bounding box has a coverage of over 80%
                coverage = get_iou(crop_box, obj_box)
                if coverage > 0.45:
                    bboxs.append([bbox_l-crop_idx, bbox_r-crop_idx, bbox_t, bbox_b]) # need to subtract crop_idx for x coord so that the box can fit.

        target["label"] = label
        target["bboxs"] = bboxs

        imgT = self.transform(img_crop)
        #print(imgT.shape, img_crop.shape)
        return (imgT, target)

    def __len__(self):
        return len(self.img_class_info)


if __name__ == "__main__":

    train_image_dir = "/projectnb/saenkog/shawnlin/object-tracking/dataset/data_tracking_image_2/training/image_02/"
    train_label_dir = "/projectnb/saenkog/shawnlin/object-tracking/dataset/data_tracking_label_2/training/label_02/"
    #test_image_dir = "/projectnb/saenkog/shawnlin/object-tracking/dataset/data_tracking_image_2/testing/image_02/"
    #test_label_dir = "/projectnb/saenkog/shawnlin/object-tracking/dataset/data_tracking_label_2/testing/label_02/"

    image_class_info = gen_kitti_frame_bins(train_image_dir, train_label_dir)
    balanced_image_class_info = balance_img_class_info(image_class_info)
    balanced_image_class_info.to_csv("./train_image_class_info.csv", sep=",")
