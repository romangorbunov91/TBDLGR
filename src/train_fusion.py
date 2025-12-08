from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader

from datasets.Bukva import Bukva
from datasets.Briareo import Briareo
from datasets.NVGestures import NVGesture
from models.model_utilizer import ModuleUtilizer
from models.temporal_fusion import GestureTransformerFusion
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils.average_meter import AverageMeter
from tensorboardX import SummaryWriter


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class GestureTrainer(object):
    """Trainer for fusion model: expects dataset to return (frames, landmarks), label."""

    def __init__(self, configer):
        self.configer = configer
        self.data_path = configer.get("data", "data_path")

        self.losses = {"train": AverageMeter(), "val": AverageMeter(), "test": AverageMeter()}
        self.accuracy = {"train": AverageMeter(), "val": AverageMeter(), "test": AverageMeter()}

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)
        self.net = None
        self.lr = None
        self.optimizer = None
        self.iters = None
        self.epoch = 0
        self.train_transforms = None
        self.val_transforms = None
        self.loss = None

        self.tbx_summary = SummaryWriter(str(Path(configer.get('checkpoints', 'tb_path'))
                                             / configer.get("dataset")
                                             / configer.get('checkpoints', 'save_name')))
        self.tbx_summary.add_text('parameters', str(self.configer).replace("\n", "\n\n"))
        self.save_iters = self.configer.get('checkpoints', 'save_iters')

        self.backbone = self.configer.get("network", "backbone")
        self.in_planes = None
        self.clip_length = self.configer.get("data", "n_frames")
        self.n_classes = self.configer.get("data", "n_classes")
        self.data_type = self.configer.get("data", "type")
        self.dataset = self.configer.get("dataset").lower()
        self.optical_flow = self.configer.get("data", "optical_flow")
        if self.optical_flow is None:
            self.optical_flow = True
        self.scheduler = None

    def init_model(self):
        if self.optical_flow is True:
            self.in_planes = 2
        elif self.data_type in ["depth", "ir"]:
            self.in_planes = 1
        else:
            self.in_planes = 3

        self.loss = nn.CrossEntropyLoss().to(self.device)

        self.net = GestureTransformerFusion(self.backbone, self.in_planes, self.n_classes,
                                            pretrained=self.configer.get("network", "pretrained"),
                                            n_head=self.configer.get("network", "n_head"),
                                            dropout_backbone=self.configer.get("network", "dropout2d"),
                                            dropout_transformer=self.configer.get("network", "dropout1d"),
                                            dff=self.configer.get("network", "ff_size"),
                                            n_module=self.configer.get("network", "n_module"),
                                            landmark_dim=63)

        self.iters = 0
        self.epoch = None
        phase = self.configer.get('phase')

        if phase == 'train':
            self.net, self.iters, self.epoch, optim_dict = self.model_utility.load_net(self.net)
        else:
            raise ValueError('Phase: {} is not valid.'.format(phase))

        if self.epoch is None:
            self.epoch = 0

        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net, self.iters)
        self.scheduler = MultiStepLR(self.optimizer, self.configer["solver", "decay_steps"], gamma=0.1)

        if optim_dict is not None:
            print("Resuming training from epoch {}.".format(self.epoch))
            self.optimizer.load_state_dict(optim_dict)

        if self.dataset == "briareo":
            Dataset = Briareo
            self.train_transforms = iaa.Sequential([
                iaa.Resize((0.85, 1.15)),
                iaa.CropToFixedSize(width=190, height=190),
                iaa.Rotate((-15, 15))
            ])
            self.val_transforms = iaa.CenterCropToFixedSize(200, 200)
        elif self.dataset == "bukva":
            Dataset = Bukva
            self.train_transforms = None
            self.val_transforms = None
        elif self.dataset == "nvgestures":
            Dataset = NVGesture
            self.train_transforms = iaa.Sequential([
                iaa.Resize((0.8, 1.2)),
                iaa.CropToFixedSize(width=256, height=192),
                iaa.Rotate((-15, 15))
            ])
            self.val_transforms = iaa.CenterCropToFixedSize(256, 192)
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        self.train_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="train", data_type=self.data_type,
                    transforms=self.train_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow),
            batch_size=self.configer.get('data', 'batch_size'), shuffle=True, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="val", data_type=self.data_type,
                    transforms=self.val_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow),
            batch_size=self.configer.get('data', 'batch_size'), shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        if self.dataset == "nvgestures":
            self.test_loader = None
        else:
            self.test_loader = DataLoader(
                Dataset(self.configer, self.data_path, split="test", data_type=self.data_type,
                        transforms=self.val_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow),
                batch_size=1, shuffle=False, drop_last=True,
                num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __train(self):
        self.net.train()
        for data_tuple in tqdm(self.train_loader, desc="Train"):
            (frames, landmarks) = data_tuple[0]
            frames = frames.to(self.device)
            landmarks = landmarks.to(self.device)
            gt = data_tuple[1].to(self.device)

            output = self.net((frames, landmarks))

            self.optimizer.zero_grad()
            loss = self.loss(output, gt.squeeze(dim=1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            predicted = torch.argmax(output.detach(), dim=1)
            correct = gt.detach().squeeze(dim=1)

            self.iters += 1
            self.update_metrics("train", loss.item(), frames.size(0),
                                float((predicted == correct).sum()) / len(correct))

    def __val(self):
        self.net.eval()
        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=str(np.random.randint(200)))):
                (frames, landmarks) = data_tuple[0]
                frames = frames.to(self.device)
                landmarks = landmarks.to(self.device)
                gt = data_tuple[1].to(self.device)

                output = self.net((frames, landmarks))
                loss = self.loss(output, gt.squeeze(dim=1))

                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach().squeeze(dim=1)

                self.iters += 1
                self.update_metrics("val", loss.item(), frames.size(0),
                                    float((predicted == correct).sum()) / len(correct))

        self.tbx_summary.add_scalar('val_loss', self.losses["val"].avg, self.epoch + 1)
        self.tbx_summary.add_scalar('val_accuracy', self.accuracy["val"].avg, self.epoch + 1)
        accuracy = self.accuracy["val"].avg
        self.accuracy["val"].reset()
        self.losses["val"].reset()

        ret = self.model_utility.save(accuracy, self.net, self.optimizer, self.iters, self.epoch + 1)
        if ret < 0:
            return -1
        elif ret > 0 and self.test_loader is not None:
            self.__test()
        return ret

    def __test(self):
        self.net.eval()
        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.test_loader, desc="Test", postfix=str(self.accuracy["test"].avg))):
                (frames, landmarks) = data_tuple[0]
                frames = frames.to(self.device)
                landmarks = landmarks.to(self.device)
                gt = data_tuple[1].to(self.device)

                output = self.net((frames, landmarks))
                loss = self.loss(output, gt.squeeze(dim=1))

                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach().squeeze(dim=1)

                self.iters += 1
                self.update_metrics("test", loss.item(), frames.size(0),
                                    float((predicted == correct).sum()) / len(correct))
        self.tbx_summary.add_scalar('test_loss', self.losses["test"].avg, self.epoch + 1)
        self.tbx_summary.add_scalar('test_accuracy', self.accuracy["test"].avg, self.epoch + 1)
        self.losses["test"].reset()
        self.accuracy["test"].reset()

    def train(self):
        for n in range(self.configer.get("epochs")):
            print("Starting epoch {}".format(self.epoch + 1))
            self.__train()
            ret = self.__val()
            if ret < 0:
                print("Got no improvement for {} epochs, current epoch is {}."
                      .format(self.configer.get("checkpoints", "early_stop"), n))
                break
            self.epoch += 1

    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()
