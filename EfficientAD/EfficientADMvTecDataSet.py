import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import random


class EfficientAD_MvTecDataset(Dataset):
    def __init__(self, path, category: str = 'leather', mode: str = 'train', size: tuple[int] = (256, 256), file_ending: str = '.npy', imageNet_path: str = "../data/ImageNet",
                 imageNet_file_ending: str = '.jpeg', preload=False, num_channels: int = 6, mean:list = None, std:list = None):
        self.path = path
        self.mode = mode
        self.file_ending = file_ending
        self.imageNet_file_ending = imageNet_file_ending
        self.imageNet_path = imageNet_path
        self.files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(os.path.join(self.path, category))
            for file in files if file.lower().endswith(self.file_ending) and mode in os.path.join(root, file)
        ]
        self.preload = preload
        if preload:
            self.images = [self.load_from_disk(fn) for fn in self.files]
        self.size = size
        if mode == 'test':
            self.gt = {}
            for file in self.files:
                if "good" not in file:
                    self.gt[file] = file.replace(self.file_ending, "_mask" + self.file_ending).replace("test", "ground_truth")
        if imageNet_path is not None:
            self.imgNet_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(imageNet_path)
                for file in files if file.lower().endswith(self.imageNet_file_ending) and mode in os.path.join(root, file)
            ]
        self.num_channels = num_channels
        self.mean = mean
        self.std = std

    def load_from_disk(self, fn):
        if 'npy' in fn:
            return np.load(fn)
        else:
            return cv.imread(fn, cv.IMREAD_COLOR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.preload:
            image = self.images[idx]
        else:
            image = self.load_from_disk(self.files[idx])
        image = (image / 255.0 - self.mean) / self.std

        if "npy" in file:
            image = image[:, :, [2, 1, 0, 5, 4, 3]] # convert from BGR to RGB
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            image = torchvision.transforms.Resize(self.size)(image)
        else:
            image = np.array(image, dtype=np.float32)
            image = cv.resize(image, (self.size[0], self.size[1]))
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        # normalize by ImageNet mean and std

        if self.mode == "train":
            if self.imageNet_file_ending == ".npy":
                imgNet_img = np.load(random.choice(self.imgNet_files))
            else:
                imgNet_img = np.array(cv.imread(random.choice(self.imgNet_files)))
            imgNet_img = np.transpose(imgNet_img, (2, 0, 1))
            imgNet_img = torch.from_numpy(imgNet_img)
            imgNet_img = torchvision.transforms.Resize(self.size)(imgNet_img)
            return {"image": image, "imgNet_img": imgNet_img}
        else:
            if "good" in file:
                return {"image": image, "label": 0, "mask": np.zeros(self.size).astype(bool), "defect": "good"}
            else:
                if "npy" in file:
                    mask = np.load(self.gt[file])
                else:
                    mask = cv.imread(self.gt[file], cv.IMREAD_GRAYSCALE)
                mask = cv.resize(mask, self.size).astype(bool)
                return {"image": image, "label": 1, "mask": mask[:, :, 0], "defect": Path(self.files[idx]).parent.name}
