from utils.config import cfg

import datetime

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100


def ds_worker_init_fn(worker_id):
    np.random.seed(datetime.datetime.now().microsecond + worker_id)


class DataContainer:
    def __init__(self, mode):
        self.dataset, self.dataloader = None, None
        self.mode = mode
        self.mode_cfg = cfg.get(self.mode.upper())

        self.create()

    def create(self):
        self.create_dataset()
        self.create_dataloader()

    @staticmethod
    def create_transform():
        transformations_final = [
            torchvision.transforms.ToTensor()
        ] + ([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-driven
        ] if not cfg.DATASET_NAME == 'MNIST' else [])

        return torchvision.transforms.Compose(transformations_final)

    def create_dataset(self):
        spatial_transform = self.create_transform()

        if cfg.DATASET_NAME == 'MNIST':
            self.dataset = MNIST(cfg.DATASET_DIR, True if self.mode == 'train' else False, spatial_transform,
                                 download=True)
        elif cfg.DATASET_NAME == 'CIFAR10':
            self.dataset = CIFAR10(cfg.DATASET_DIR, True if self.mode == 'train' else False, spatial_transform,
                                   download=True)
        elif cfg.DATASET_NAME == 'CIFAR100':
            self.dataset = CIFAR100(cfg.DATASET_DIR, True if self.mode == 'train' else False, spatial_transform,
                                    download=True)
        else:
            raise NotImplementedError

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.mode_cfg.BATCH_SIZE,
                                     shuffle=self.mode_cfg.SHUFFLE,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=True,
                                     worker_init_fn=ds_worker_init_fn,
                                     )
