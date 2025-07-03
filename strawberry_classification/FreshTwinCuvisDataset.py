import torchvision
from torchvision.transforms import v2

from torch.utils.data import Dataset
import cuvis
import os
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from functools import partial
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from gpu_pca import IncrementalPCAonGPU as t_pca
import json
class FreshTwinCuvisDataset(Dataset):
    def __init__(self, root_dir: Path, mode: str = 'train', in_channels: int = 38, mean: list = None, std: list = None, max_img_shape: int = 2000, normalize: bool = False, white_path: str = None, dark_path: str = None) :
        self.root_dir = root_dir
        self.mode = mode
        with open("../good_images.json", "r") as f:
            good_images = json.load(f)
        self.file_paths = [
            path
            for path in Path(self.root_dir).glob("*.cu3s")
            if
            "_28_04" not in path.name
            #and
            #("_20" in path.name or "_21" in path.name)
        ]

        self.images = [[file_path, index]
                       for file_path in self.file_paths
                       for index in range(len(cuvis.SessionFile(file_path)))]
        self.in_channels = in_channels
        self.masks = {}
        for file_path in self.file_paths:
            self.masks[file_path] = file_path.parent/"masks"/(file_path.stem + "_0000_Strawberry_swir_fasterRGB_mask.npy")
        self.mean = mean
        self.std = std
        self.max_img_shape = max_img_shape
        self.normalize = normalize
        self.proc = None
        self.white_path = white_path
        self.dark_path = dark_path

        self.pca = t_pca(n_components=in_channels)
        for img_path in tqdm(self.images, desc="training PCA"):
            sess = cuvis.SessionFile(img_path[0])
            mesu = sess.get_measurement(img_path[1])
            if "cube" not in mesu.data:
                if self.proc is None:
                    # create processing context only once if there are session files without cubes
                    self.proc = cuvis.ProcessingContext(sess)
                    self.proc.set_reference( cuvis.SessionFile(self.white_path).get_measurement(0),cuvis.ReferenceType.White)
                    self.proc.set_reference( cuvis.SessionFile(self.dark_path).get_measurement(0),cuvis.ReferenceType.Dark)
                    self.proc.processing_mode = cuvis.ProcessingMode.Reflectance
                mesu = self.proc.apply(mesu)
            cube = torch.from_numpy(mesu.data["cube"].array).to("cuda")
            cube = cube.permute(2, 0, 1)  # transpose from H x W x C to C x H x W for torch
            cube = cube / 10000  # 100% reflectance equals 10000, we divide by that to make 100% reflectance equal 1
            cube = cube.permute(1, 2, 0).reshape(-1, 38)  # reshape for PCA
            self.pca.partial_fit(cube)
        self.proc = None
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx][0]
        file_name = file_path.name.split("_")
        sess = cuvis.SessionFile(file_path)
        mesu = sess.get_measurement(self.images[idx][1])
        if "cube" not in mesu.data:
            if self.proc is None:
                # create processing context only once if there are session files without cubes
                self.proc = cuvis.ProcessingContext(sess)
                self.proc.set_reference(cuvis.SessionFile(self.white_path).get_measurement(0), cuvis.ReferenceType.White)
                self.proc.set_reference(cuvis.SessionFile(self.dark_path).get_measurement(0), cuvis.ReferenceType.Dark)
                self.proc.processing_mode = cuvis.ProcessingMode.Reflectance
            mesu = self.proc.apply(mesu)
        cube = torch.from_numpy(mesu.data["cube"].array).to("cuda")
        cube = cube.permute(2, 0, 1)  # transpose from H x W x C to C x H x W for torch
        cube = cube / 10000 # 100% reflectance equals 10000, we divide by that to make 100% reflectance equal 1
        rgb = torch.zeros(3,200,200)
        # rgb[0] = torch.mean(cube[2:7], dim=0)
        # rgb[1] = torch.mean(cube[7:17], dim=0)
        # rgb[2] = torch.mean(cube[23:28], dim=0)
        rgb[0] = cube[4]
        rgb[1] = cube[12]
        rgb[2] = cube[25]
        # R 2:7 G 7:17 B 23:28
        cube = cube.permute(1, 2, 0).reshape(-1, 38) # reshape for PCA


        cube = self.pca.transform(cube)
        cube = cube.reshape(200, 200, self.in_channels).permute(2, 0, 1).type(torch.float16) # reshape back
        if self.normalize:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            cube = torchvision.transforms.Resize(size=self.max_img_shape - 1, max_size=self.max_img_shape)(cube)
        #cube = torch.unsqueeze(cube, 0)
        mask = torch.tensor(np.load(self.masks[file_path]), device="cuda")
        label = 0
        if mask.max() > 1: #TODO check if we want to label like this
            label = 1
        return {"image": cube, "mask": mask, "label": label, "number": file_name[1], "side": file_name[2], "day": file_name[3], "rgb_image": rgb}
