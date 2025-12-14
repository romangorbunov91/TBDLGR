import os
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

# Import Datasets
from datasets.Bukva import Bukva
from models.model_utilizer import ModuleUtilizer

# Import Model
from models.temporal import GestureTransoformer

# Import Utils
from tqdm import tqdm
from utils.average_meter import AverageMeter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class GestureTest(object):
    """Gesture Recognition Test class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        train_loader (torch.utils.data.DataLoader): Train data loader variable
        val_loader (torch.utils.data.DataLoader): Val data loader variable
        test_loader (torch.utils.data.DataLoader): Test data loader variable
        net (torch.nn.Module): Network used for the current procedure
        lr (int): Learning rate value
        optimizer (torch.nn.optim.optimizer): Optimizer for training procedure
        iters (int): Starting iteration number, not zero if resuming training
        epoch (int): Starting epoch number, not zero if resuming training
        scheduler (torch.optim.lr_scheduler): Scheduler to utilize during training

    """

    def __init__(self, configer):
        self.configer = configer

        self.data_path = configer.get("data", "data_path")      #: str: Path to data directory

        # Train val and test accuracy
        self.accuracy = AverageMeter()

        # DataLoaders
        self.data_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None

        # Training procedure
        self.transforms = None

        # Other useful data
        self.backbone = self.configer.get("network", "backbone")    #: str: Backbone type
        self.in_planes = None                                       #: int: Input channels
        self.clip_length = self.configer.get("data", "n_frames")    #: int: Number of frames per sequence
        self.n_classes = self.configer.get("data", "n_classes")     #: int: Total number of classes for dataset
        self.data_type = self.configer.get("data", "type")          #: str: Type of data (rgb, depth, ir, leapmotion)
        self.dataset = self.configer.get("dataset").lower()         #: str: Type of dataset
        self.optical_flow = self.configer.get("data", "optical_flow")
        if self.optical_flow is None:
            self.optical_flow = True

    def init_model(self):
        """Initialize model and other data for procedure"""

        if self.optical_flow is True:
            self.in_planes = 2
        elif self.data_type in ["depth", "ir"]:
            self.in_planes = 1
        else:
            self.in_planes = 3

        # Selecting correct model and normalization variable based on type variable
        self.net = GestureTransoformer(self.backbone, self.in_planes, self.n_classes,
                                       pretrained=self.configer.get("network", "pretrained"),
                                       n_head=self.configer.get("network", "n_head"),
                                       dropout_backbone=self.configer.get("network", "dropout2d"),
                                       dropout_transformer=self.configer.get("network", "dropout1d"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module")
                                       )

        self.net, _, _, _ = self.model_utility.load_net(self.net)

        # Selecting Dataset and DataLoader
        if self.dataset == "bukva":
            Dataset = Bukva
            self.transforms = iaa.Noop()
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        # Setting Dataloaders
        self.data_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="test", data_type=self.data_type,
                    transforms=self.transforms, n_frames=self.clip_length,
                    optical_flow=self.optical_flow),
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """Testing function."""
        self.net.eval()
        correct = 0
        total = 0
        all_preds = []   # To store all predictions
        all_labels = []  # To store all ground truth labels

        with torch.no_grad():
            for data_tuple in tqdm(self.data_loader, desc="Test"):
                inputs, gt = data_tuple[0].to(self.device), data_tuple[1].to(self.device)

                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)
                labels = gt.flatten()  # Ensure shape [B]

                # Accumulate predictions and labels.
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                # Accuracy tracking
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Plot and save to PDF
        fig, ax = plt.subplots(figsize=(16, 12))  # Adjust size if needed
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix (TEST mean accuracy: {accuracy:.4f})")

        # Save as PDF
        # Create destination folder.
        dir_path = Path(self.configer.get('scores', 'save_dir'))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        output_path = dir_path/ f"TEST_acc_{accuracy:.4f}.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)  # Free memory
        print(f"Confusion matrix saved to {output_path}")
 
    def test(self):
        self.__test()

    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()