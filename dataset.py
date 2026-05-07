"""
Author: Rahul Tole
Custom dataset with augmented imagess
"""

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class MarsDataset(Dataset):
    def __init__(self, image_dir, label_dir, transformation=None, mode=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transformation
        self.mode = mode
        self.samples = []

        labels = os.listdir(label_dir)
        for name in labels:
            # Switches between train & test images
            if mode == "train":
                image_path = os.path.join(self.image_dir, name.replace(".npz", ".jpg"))
            else:
                image_path = os.path.join(self.image_dir, name.replace("_merged", "").replace(".png", ".jpg"))
            label_path = os.path.join(self.label_dir, name)

            if os.path.exists(image_path) and os.path.exists(label_path):
                self.samples.append((image_path, label_path))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        image = np.array(Image.open(image_path).convert("L"))
        # Switches between train & test images
        if self.mode == "train":
            label = np.load(label_path)['label']
        else:
            label = np.array(Image.open(label_path).convert("L"), dtype=np.int16)

        # applies transformations
        if self.transform is not None:
            augmentation = self.transform(image=image, mask=label)
            image = augmentation['image']
            label = augmentation['mask'].long()

        return image, label


