import matplotlib.pyplot as plt
import numpy as np

from src.adversarial_attacks.utils import config


def plot_images(data_path: str) -> None:
    """
    Plot 5 images and display their classes

    :param data_path: Path containing images to plot
    :return:
    """
    n_samples = config["n_samples"]
    X, y = np.load(data_path, allow_pickle=True)
    random_img = np.random.choice(len(X), config["n_samples"], replace=False)

    plt.figure(figsize=tuple(config["fig_dims"]))

    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.axis("off")
        plt.imshow(X[random_img[i]])
        plt.title(config["classes"][np.argmax(y[random_img[i]])], fontsize=20)

    plt.suptitle("Example Images", fontsize=25)
    plt.show()
