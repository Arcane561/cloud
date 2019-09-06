"""
Fastai style masks need to be created.
"""

from helper_functions import create_mask

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
print(__file__)

for ind, img in enumerate(os.listdir(TRAIN_DIR)):
    if ind % 100 == 0: print(ind)

    if not img.endswith("jpg"):
        print(f"{img} not a valid image")
        continue
        
    mask = create_mask(img)

    # Convert numpy array to PIL.Image.Image
    mask_img = Image.fromarray(mask, mode="L")

    # Save as png in train_lables
    mask_img.save(PROJ_DIR / "data" / "raw" / "train_labels" / f"{img}")
