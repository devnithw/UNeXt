import os
import cv2
import numpy as np
import torch
import torch.utils.data

class BUSIDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Breast Ultrasound Images Dataset (BUSI). Data should be included
    as the following structure.
    data/
    ├── images/
    │   ├── 001.png
    │   ├── 002.png
    ├── masks/
    │   ├── 0/
    │   |   ├── 001.png
    |   |   ├── 002.png
    """
    
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get image ID
        img_id = self.img_ids[idx]

        # Load image from Open CV
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # Get masks (in our case only 1)
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        # Apply transform
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Normalize image
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        # Give image, mask and its id
        return img, mask, {'img_id': img_id}