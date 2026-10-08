"""Fashion-MNIST loading, normalization, and Lightning data-module support."""
import gzip
import os

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

MEAN, STD = 0.2860, 0.3530
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion_mnist(root="data"):
    """Load Fashion-MNIST through torchvision or local raw IDX files."""
    try:
        from torchvision import datasets

        train = datasets.FashionMNIST(root, train=True, download=True)
        test = datasets.FashionMNIST(root, train=False, download=True)
        return train.data, train.targets, test.data, test.targets
    except ImportError:
        raw = os.path.join(root, "FashionMNIST", "raw")

        def read(name, offset):
            with gzip.open(os.path.join(raw, name + ".gz")) as file:
                return torch.from_numpy(
                    np.frombuffer(file.read(), np.uint8, offset=offset).copy()
                )

        train_images = read("train-images-idx3-ubyte", 16).view(-1, 28, 28)
        train_targets = read("train-labels-idx1-ubyte", 8).long()
        test_images = read("t10k-images-idx3-ubyte", 16).view(-1, 28, 28)
        test_targets = read("t10k-labels-idx1-ubyte", 8).long()
        return train_images, train_targets, test_images, test_targets


def preprocess(images):
    """Normalize image tensors using the experiment's fixed dataset statistics."""
    return ((images.float() / 255 - MEAN) / STD).unsqueeze(1)


class FashionMNISTDataModule(pl.LightningDataModule):
    """Provide deterministic train and evaluation loaders to Lightning."""

    def __init__(
        self,
        train_images,
        train_targets,
        eval_images,
        eval_targets,
        batch_size=128,
        seed=0,
        eval_batch_size=2000,
    ):
        super().__init__()
        self.train_images = train_images
        self.train_targets = train_targets
        self.eval_images = eval_images
        self.eval_targets = eval_targets
        self.batch_size = batch_size
        self.seed = seed
        self.eval_batch_size = eval_batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.train_images, self.train_targets)
        self.eval_dataset = TensorDataset(self.eval_images, self.eval_targets)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=4,  # Adjust this based on your system's capabilities
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4,  # Adjust this based on your system's capabilities
        )