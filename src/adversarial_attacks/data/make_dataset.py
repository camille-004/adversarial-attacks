import glob
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import click
import numpy as np

from src.adversarial_attacks.logger import logger
from src.adversarial_attacks.utils import config

log = logger.setup_module_level_logger(__name__, file_name=config["data_log"])  # type: ignore


def unpickle_cifar(batch_id: Union[str, int]) -> Tuple[np.ndarray, List]:
    """
    Unpickle a given batch of the CIFAR-10 dataset

    :param batch_id: Batch ID
    :return: NumPy array of features and list of labels of one CIFAR-10 batch
    """
    path = os.path.join(Path(__file__).parents[3], config["raw_data_path"])

    if batch_id == "test":
        file_name = "/test_batch"
    else:
        file_name = f"/data_batch_{batch_id}"

    with open(path + file_name, "rb") as f:
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


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize input images.

    :param x: NumPy array of input images (each image is [32, 32, 3])
    :return: Normalized image
    """
    return (x - np.min(x)) / (x - np.max(x))


def one_hot_encode(x: List) -> np.ndarray:
    """
    One-hot-encode a list of labels

    :param x: Input list of labels from one batch
    :return: One-hot-encoded NumPy array
    """
    encoded = np.zeros((len(x), config["n_labels"]))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded


def _preprocess_data(
    _features: np.ndarray, _labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess input features and labels.

    :param _features: Input features.
    :param _labels: Input labels.
    :return: Tuple of preprocessed featurs and labels.
    """
    return normalize(np.array(_features)), one_hot_encode(_labels)


def preprocess_data(n_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load n_batches batches, normalize features, and one-hot-encode labels.

    :param n_batches: Number of batches to load
    :return: Preprocessed features and labels
    """
    assert n_batches <= config["n_batches"], "Specified batches must be <= 5."
    features = []
    labels = []

    for i in range(1, n_batches + 1):
        feature_batch, label_batch = unpickle_cifar(i)
        features.extend(feature_batch)
        labels.extend(label_batch)

    return _preprocess_data(features, labels)


def save_data(features: np.ndarray, labels: np.ndarray, path: str) -> None:
    """
    Save features and labels as a pickle file.

    :param features: Features to save.
    :param labels: One-hot-encoded labels to save.
    :param path: Output save path.
    :return:
    """
    assert (
        labels.shape[1] == config["n_labels"]
    ), "Labels must be one-hot-encoded."
    pickle.dump((features, labels), open(path, "wb"))


@click.command()
@click.argument("n_batches", type=click.INT)
@click.argument("output_file_name", type=click.Path())
@click.option(
    "--prepare_test",
    is_flag=True,
    default=False,
    help="Whether to prepare test data.",
)
def main(n_batches: int, output_file_name: str, prepare_test: bool) -> None:
    """
    Run data processing scripts to turn raw data into processed data to be saved in processed data folder.
    """
    log.info("Making final dataset from raw data")
    out_path = os.path.join(config["processed_data_path"], output_file_name)

    features, labels = preprocess_data(n_batches)

    files = glob.glob(f"{config['processed_data_path']}*")
    for f in files:  # Clear all files in processed directory before saving.
        os.remove(f)

    save_data(features, labels, out_path)

    if prepare_test:
        log.info(f"Saved training data: {out_path}")
        test_features, test_labels = unpickle_cifar("test")
        test_features, test_labels = _preprocess_data(
            test_features, test_labels
        )
        save_data(test_features, labels, out_path + "_test")
        log.info(f"Saved test data: {out_path + '_test'}")

    log.info(f"Data pickle saved in {config['processed_data_path']}")


if __name__ == "__main__":
    main()
