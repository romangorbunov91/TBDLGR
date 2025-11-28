import torch
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from torch.utils.data.dataset import Dataset

from datasets.utils.normalize import normalize

class Bukva(Dataset):
    """Bukva Dataset class"""
    def __init__(self, configer, path, split="train", data_type=None, transforms=None, n_frames=None, optical_flow=False):
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

        print("Loading Bukva {} annotations...".format(split.upper()), end=" ")
        
        annotations_file_name = 'annotations_cut.csv'
        csv_path = self.dataset_path / annotations_file_name
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Не найден файл аннотаций: {csv_path}")
        
        df = pd.read_csv( csv_path )
        # 2. Фильтрация по сплиту (train/val/test)
        # В CSV колонка называется 'split'
        data_df = df[df['split'] == split].reset_index(drop=True)  

        # Превращаем DataFrame в список словарей для удобства (как в Briareo self.data)
        fixed_data = list()
        for _, row in data_df.iterrows():
            filenames = ['frames/' + str(row['attachment_id']) + f'/frame_{i:03d}.jpg' for i in range(n_frames)]
            record = {
                'label': int(row['label_encoded']), # Числовая метка
                'data': filenames # Имя папки с кадрами
            }
            fixed_data.append(record)
                
        self.data = np.array(fixed_data)
        print(f"done. Found {len(self.data)} {split.upper()}-samples in '{annotations_file_name}'")

        # Проверяем количество frames.
        img_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        for _, row in data_df.iterrows():
            frame_dir = 'frames/' + str(row['attachment_id'])
            files = [fl for fl in Path(str(self.dataset_path / frame_dir)).iterdir() if fl.is_file() and fl.suffix.lower() in img_ext]
            if len(files) < n_frames:
                raise ValueError(f"{str(frame_dir)}: only {len(files)} image(s) found (must be >= n_frames={n_frames})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]['data']
        label = self.data[idx]['label']

        clip = list()
        for p in paths:
            img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            clip.append(img)

        clip = np.array(clip).transpose(1, 2, 3, 0)
        clip = normalize(clip)

        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
        label = torch.LongTensor(np.asarray([label]))
        return clip.float(), label
