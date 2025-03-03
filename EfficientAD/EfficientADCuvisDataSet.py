import torchvision
from torch.utils.data import Dataset
import cuvis
import os
from cuvis.cuvis_types import ProcessingMode
import numpy as np
import cv2 as cv
import random
import torch
from torchvision.transforms import v2
from functools import partial
from pathlib import Path


class EfficientADCuvisDataSet(Dataset):
    def __init__(self, path: str = "data/cubes", proc_mode: int = 4, size: tuple[int] = (256, 256), mode: str = "train", imageNet_path: str = "../data/ImageNet_6_channel",
                 imageNet_file_ending: str = '.npy', in_channels: int = 6, mean: list = None, std: list = None, normalize: bool = True, max_img_shape: int = 1500):
        self.path = path
        self.size = size
        self.mode = mode
        self.proc_mode = proc_mode
        self.imageNet_file_ending = imageNet_file_ending
        self.imageNet_path = imageNet_path
        self.file_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.path)
            for file in files if file.lower().endswith(".cu3s")
        ]
        self.in_channels = in_channels
        self.images = [[file_path, index]
                       for file_path in self.file_paths
                       for index in range(len(cuvis.SessionFile(file_path)))]

        if imageNet_path is not None:
            self.imgNet_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(imageNet_path)
                for file in files if file.lower().endswith(self.imageNet_file_ending) and mode in os.path.join(root, file)
            ]
        if mode == 'test':
            self.gt = {}
            for file_path in self.file_paths:
                if "_ok_ok_" not in file_path:
                    # TODO: refactor this
                    self.gt[file_path] = file_path.replace(".cu3s", "_mask.png").replace("test", "ground_truth")

        self.transform = v2.Compose([
            v2.Lambda(torch.as_tensor),
            v2.ToDtype(torch.float32, scale=False),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.Lambda(partial(torch.rot90, k=0, dims=(-2, -1))),  # rotate 0 deg
                v2.Lambda(partial(torch.rot90, k=1, dims=(-2, -1)))  # rotate 90 deg
            ]),
        ])

        self.proc = None
        self.mean = mean
        self.std = std
        self.max_img_shape = max_img_shape
        self.normalize = normalize
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx][0]
        sess = cuvis.SessionFile(file_path)
        mesu = sess.get_measurement(self.images[idx][1])
        if "cube" not in mesu.data:
            if self.proc is None:
                # create processing context only once if there are session files without cubes
                self.proc = cuvis.ProcessingContext(sess)
                self.proc.processing_mode = ProcessingMode.Raw

            mesu = self.proc.apply(mesu)
        cube = mesu.data["cube"].array
        if self.normalize:
            cube = (cube - self.mean) / self.std
        cube = np.transpose(cube, (2, 0, 1))  # transpose from H x W x C to C x H x W for torch
        cube = cube.astype(np.float32)
        if self.in_channels == 3:
            cube = cube[[2, 1, 0]]
        else:
            # BGR -> RGB
            cube = cube[[2, 1, 0, 3, 4, 5]]
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            cube = torch.from_numpy(cube)
            cube = torchvision.transforms.Resize(size=self.max_img_shape - 1, max_size=self.max_img_shape)(cube)
        if self.mode == "train":

            if self.imageNet_file_ending == ".npy":
                imgNet_img = np.load(random.choice(self.imgNet_files))
            else:
                imgNet_img = np.array(cv.imread(random.choice(self.imgNet_files)))
            imgNet_img = np.transpose(imgNet_img, (2, 0, 1))  # transpose from H x W x C to C x H x W for torch
            imgNet_img = (imgNet_img / 255).astype(np.float32)
            imgNet_img = torch.from_numpy(imgNet_img)
            if imgNet_img.shape[1] > 1000 or imgNet_img.shape[2] > 1000 or imgNet_img.shape[1] < 256 or imgNet_img.shape[2] < 256:
                imgNet_img = torchvision.transforms.Resize(size=500, max_size=1000)(imgNet_img)

            return self.transform({"image": cube, "imgNet_img": imgNet_img})
        else:
            if "_ok_ok_" in file_path:
                return {"image": cube, "label": 0, "mask": torch.zeros(cube.shape[-2:], dtype=torch.bool), "defect": "good"}
            else:
                defect = Path(file_path).parent.name
                if os.path.exists(self.gt[file_path]):
                    mask = cv.imread(self.gt[file_path], cv.IMREAD_GRAYSCALE)
                    mask = mask > 0
                    mask = torch.from_numpy(mask)
                else:
                    mask = torch.zeros(cube.shape[-2:], dtype=torch.bool)
                return {"image": cube, "label": 1, "mask": mask, "defect": defect}
