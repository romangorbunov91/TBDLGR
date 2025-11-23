import os
import math
import torch
from pathlib import Path

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

from datasets.utils.normals import normals_multi
from datasets.utils.normalize import normalize
from datasets.utils.optical_flow import dense_flow

from datasets.utils.utils_briareo import from_json_to_list


class Bukva(Dataset):
    """Bukva Dataset class"""
    def __init__(self, configer, path, split="train", transforms=None):
        """Constructor method for Bukva Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data

        """
        super().__init__()

        self.dataset_path = Path(path)
        self.split = split

        self.transforms = transforms

        print("Loading Bukva {} dataset...".format(split.upper()), end=" ")

        self.data = np.loadtxt(self.dataset_path / "annotations.csv", delimiter=',')
        print("done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]['data']
        label = self.data[idx]['label']

        clip = list()
        for p in paths:
            img = cv2.imread(str(self.dataset_path /'frames'/ p), cv2.IMREAD_COLOR)
            img = np.expand_dims(img, axis=2)
            clip.append(img)

        clip = np.array(clip).transpose(1, 2, 3, 0)
        clip = normalize(clip)

        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
        label = torch.LongTensor(np.asarray([label]))
        return clip.float(), label
