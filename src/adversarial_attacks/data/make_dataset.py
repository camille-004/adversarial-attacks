import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import yaml  # type: ignore

from src.adversarial_attacks.logger import logger

log = logger.setup_module_level_logger(__name__, file_name="data.log")  # type: ignore

CONFIG_PATH = "../config/"


def load_config(f_name: str) -> Dict:
    """
    Retrieve configurations from a given file.

    :param f_name: Name of chosen config file.
    :return: Dictionary of YAML contents
    """
    with open(os.path.join(CONFIG_PATH, f_name)) as f:
        _config = yaml.safe_load(f)

    return _config


config = load_config("config.yml")


def unpickle_cifar(batch_id: int) -> Tuple[np.ndarray, List]:
    """
    Unpickle a given batch of the CIFAR-10 dataset

    :param batch_id: Batch ID
    :return: NumPy array of features and list of labels of one CIFAR-10 batch
    """
    path = os.path.join(Path(__file__).parents[3], config["raw_data_path"])
    with open(path + f"/data_batch_{batch_id}", "rb") as f:
        batch = pickle.load(f, encoding="latin1")

    features = (
        batch["data"]
        .reshape(
            (len(batch["data"]), 3, config["image_dim"], config["image_dim"])
        )
        .transpose(0, 2, 3, 1)
    )
    labels = batch["labels"]

    log.info(f"Unpickled batch {batch_id}")

    return features, labels


@click.command()
@click.argument("batch_id", type=click.INT)
@click.argument("output_filepath", type=click.Path())
def main(batch_id: int, output_filepath: str) -> None:
    """
    Run data processing scripts to turn raw data into processed data to be saved in processed data folder

    :param batch_id: CIFAR-10 batch
    :param output_filepath: File path in which to save processed data
    :return:
    """
    log.info("Making final dataset from raw data")

    # Call data processing functions here


if __name__ == "__main__":
    main()
