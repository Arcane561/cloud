from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

import numpy as np
from PIL import Image

from pathlib import Path
import os
import random

PROJ_DIR = Path(__file__).parent.parent
TRAIN_DIR = PROJ_DIR / "data" / "raw" / "train_images"
train_labels = PROJ_DIR / "data" / "raw" / "train.csv"
print(PROJ_DIR)

# Read encoded labels
with open(train_labels, "r") as f:
    y_train = f.read()

y_train = y_train.split("\n")

# Convert labels to dict
y_label_dict = {}
for img in y_train[1:-1]:
    y_label_dict[img.split(",")[0]] = img.split(",")[-1]

# Codes for cloud class, to be used for mask creation
cloud_class = ["Fish", "Flower", "Gravel", "Sugar"]
cloud_class_code = {x: cloud_class.index(x) + 1 for x in cloud_class}

# Reading one image for reference and obtaining shape info
img_f = "0011165.jpg" #random.choice(os.listdir(TRAIN_DIR))
img = open_image(TRAIN_DIR / img_f)

def create_mask(img_file):
    "Given a image and class show the image with labeled pixels"

    mask = np.zeros(img.data.shape[1] * img.data.shape[2], dtype=np.int64)

    for cld_clss in cloud_class_code:

        if len(y_label_dict[f"{img_file}_{cld_clss}"]) != 0:

            mask_encoded = y_label_dict[f"{img_file}_{cld_clss}"]
            mask_encoded = mask_encoded.split(" ")

            for i in range(int(len(mask_encoded) / 2)):
                i_start = int(mask_encoded[2 * i])
                running_len = int(mask_encoded[2 * i + 1])

                mask[i_start : i_start + running_len] = cloud_class_code[cld_clss]

    mask = mask.reshape((img.data.shape[1], img.data.shape[2]))

    return mask

x = create_mask(img_f)

print("Done")

