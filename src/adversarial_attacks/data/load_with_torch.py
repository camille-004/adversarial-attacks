import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src.adversarial_attacks.logger import logger
from src.adversarial_attacks.utils import config

log = logger.setup_module_level_logger(__name__, file_name=config["data_log"])  # type: ignore

BATCH_SIZE = config["torch_data_batch_size"]
CLASSES = config["classes"]


def get_CIFAR10() -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    """
    Get the CIFAR10 dataset from torchvision and return the DataLoaders.

    :return: Train and test DataLoaders.
    """
    root = os.path.join(
        Path(__file__).parents[3], config["external_data_path"]
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    log.info("From torchvision: Loaded training data")
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    log.info("From torchvision: Loaded test data")

    return train_loader, test_loader


def show_images(data_loader: torch.utils.data.DataLoader) -> None:
    """
    Plot example images from a random batch.

    :param data_loader: DataLoader from which to get images.
    :return:
    """

    def imshow(img: torch.Tensor) -> None:
        img = img / 2 + 0.5
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    data_iter = iter(data_loader)
    images, labels = data_iter.next()

    imshow(torchvision.utils.make_grid(images))

    print(" ".join(f"{CLASSES[labels[j]]:5s}" for j in range(BATCH_SIZE)))
