# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:42:59 2019

@author: joshipan

Modifying according to 
https://forums.fast.ai/t/how-to-load-multiple-classes-of-rle-strings-from-csv-severstal-steel-competition/51445
"""

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import os
import random
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

PROJ_DIR = Path(__file__).parent.parent
path_img = PROJ_DIR / "data" / "raw" / "train_images"
path_test_img = PROJ_DIR / "data" / "raw" / "test_images"
path_train_labels = PROJ_DIR / "data" / "raw" / "train_labels"
print(PROJ_DIR)

fnames = get_image_files(path_img)
img_f = fnames[10]
img = open_image(img_f)
img.show(figsize=(5, 5))

# Preparing labels codes
cloud_class = ["Fish", "Flower", "Gravel", "Sugar"]
# Classes must be 0 indexed for pytorch to work. Background gets 0 
cloud_class_code = {x: cloud_class.index(x) + 1 for x in cloud_class}
print(cloud_class_code)

# Reading run-length-encoded labels
df_labels = pd.read_csv(PROJ_DIR / "data" / "raw" / "train.csv")
df_labels.head()


def rle_2_png(img):
    """
    Takes the name of the image, creates the mask will all
    labels available and saves the mask as png
    """
    # If img lacks 'jpg' extension add it
    if not img.endswith("jpg"):
        img += ".jpg"

    # Filter labels dataframe
    df_temp = df_labels[df_labels.Image_Label.apply(lambda x: img in x)]

    # Drop rows with missing lables
    df_temp = df_temp[~df_temp.EncodedPixels.isna()]

    if len(df_temp) == 0:
        raise ValueError(f"{img} has no labels existing.")

    # Create combined mask for each class of clouds
    for ind in range(len(df_temp)):
        cloud_cls = df_temp.iloc[ind].Image_Label.split("_")[-1]

        mask = open_mask_rle(df_temp.iloc[ind].EncodedPixels, (1400, 2100))

        if ind == 0:
            mask_complete = mask.data
        else:
            mask_complete[mask.data != 0] = cloud_class_code[cloud_cls]

    #mask_complete = torch.tensor(mask_complete, dtype=torch.uint8)
    mask_complete = ImageSegment(mask_complete.data.permute(0,2,1))
    mask_complete.save(
        PROJ_DIR / "data" / "raw" / "train_labels" / f"{img.split('.')[0]}.png"
    )


for ind, fn in enumerate(fnames):
    if ind % 50 == 0:
        print(ind)
    rle_2_png(fn.stem)