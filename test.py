import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

#from albumentations.augmentations import transforms
#from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
#from albumentations import RandomRotate90,Resize
from dataset import BUSIDataset

# imaport data from BUSIDataset and plot samples

NUM_CLASSES = 1
BATCH_SIZE = 8
NUM_WORKERS = 0

# image path
data_path = r"D:\Projects\UNeXt\datasets\busi"

# read the data
# Get image IDs
img_ids = glob.glob(os.path.join(data_path, 'images', '*' + '.png'))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

# Split the data
train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

# Completet augmentations later

train_dataset = BUSIDataset(
    img_ids=train_img_ids,
    img_dir=os.path.join(data_path, 'images'),
    mask_dir=os.path.join(data_path, 'masks'),
    img_ext='.png',
    mask_ext='.png',
    num_classes=NUM_CLASSES,
    transform=None)

val_dataset = BUSIDataset(
    img_ids=val_img_ids,
    img_dir=os.path.join(data_path, 'images'),
    mask_dir=os.path.join(data_path, 'masks'),
    img_ext='.png',
    mask_ext='.png',
    num_classes=NUM_CLASSES,
    transform=None)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False)

# load and plot sample data from dataset

def plot_sample_data(img, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[0], cmap='gray')
    ax[1].imshow(mask[0], cmap='gray')
    plt.show()


for i in range(5):
    img, mask, _ = train_dataset[i]
    plot_sample_data(img, mask)
