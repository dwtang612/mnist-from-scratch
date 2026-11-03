import urllib.request
import gzip
import os
import numpy as np

DATA_DIRECTORY = "data"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download():
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

    for filename in FILES.values():
        filepath = os.path.join(DATA_DIRECTORY, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}.")
            urllib.request.urlretrieve(BASE_URL + filename, filepath)
        else:
            print(f"{filename} already exists, skipping")


def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        # First 16 bytes are metadata: magic number, count, rows, cols
        magic = int.from_bytes(f.read(4), "big")
        count = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        # The rest is pixel data, one byte per pixel, 0-255
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count, rows * cols)


def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")  # magic number (skip)
        _ = int.from_bytes(f.read(4), "big")  # count (skip)
        return np.frombuffer(f.read(), dtype=np.uint8)


if __name__ == "__main__":
    download()

    train_images = load_images(os.path.join(DATA_DIRECTORY, FILES["train_images"]))
    train_labels = load_labels(os.path.join(DATA_DIRECTORY, FILES["train_labels"]))
    test_images = load_images(os.path.join(DATA_DIRECTORY, FILES["test_images"]))
    test_labels = load_labels(os.path.join(DATA_DIRECTORY, FILES["test_labels"]))

    print(
        f"\nTraining set: {train_images.shape[0]} images, each {train_images.shape[1]} pixels"
    )
    print(
        f"Test set: {test_images.shape[0]} images, each {test_images.shape[1]} pixels"
    )
    print(f"Labels range: {train_labels.min()} to {train_labels.max()}")
    print(f"Pixel value range: {train_images.min()} to {train_images.max()}")
    print(
        f"Test labels: {test_labels.shape[0]} labels, range {test_labels.min()} to {test_labels.max()}"
    )
