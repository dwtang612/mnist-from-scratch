import matplotlib.pyplot as plt
from download_data import load_images, load_labels, DATA_DIRECTORY, FILES
import numpy as np
import os


train_images = load_images(os.path.join(DATA_DIRECTORY, FILES["train_images"]))
train_labels = load_labels(os.path.join(DATA_DIRECTORY, FILES["train_labels"]))
# test_images = load_images(os.path.join(DATA_DIRECTORY, FILES["test_images"]))
# test_labels = load_labels(os.path.join(DATA_DIRECTORY, FILES["test_labels"]))

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)  # 5 rows, 5 cols, position i+1
    image = train_images[i].reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.title(train_labels[i])
    plt.axis("off")  # hide the pixel coordinates

plt.tight_layout()
plt.show()
