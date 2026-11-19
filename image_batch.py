from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np


class BaseBatch(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def display_metadata(self):
        pass


class LoggingMixin:
    def log_operation(self, message):
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}")


class ImageBatch(LoggingMixin, BaseBatch):
    def __init__(self, images, labels):
        images = np.asarray(images)
        labels = np.asarray(labels)
        if images.ndim != 4:
            raise ValueError("images must have shape (N, H, W, C)")
        if labels.ndim != 1:
            raise ValueError("labels must have shape (N,)")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("Number of images must match number of labels")
        self.images = images
        self.labels = labels
        self.log_operation(f"Created ImageBatch of size {len(self)}.")

    def __len__(self):
        return self.images.shape[0]

    def __add__(self, other):
        # Defense: H, W, C are checked first so mismatched batches fail with a
        # clear ValueError. Images and labels are concatenated along axis 0 in
        # the same order (self, then other), so each label stays aligned with
        # its image. A new ImageBatch is returned and neither input is changed.
        if not isinstance(other, ImageBatch):
            return NotImplemented
        if self.images.shape[1:] != other.images.shape[1:]:
            raise ValueError("Image dimensions (H, W, C) do not match")
        self.log_operation(
            f"Concatenating two batches of size {len(self)} and {len(other)}."
        )
        images = np.concatenate((self.images, other.images), axis=0)
        labels = np.concatenate((self.labels, other.labels), axis=0)
        return ImageBatch(images, labels)

    def display_metadata(self):
        values, counts = np.unique(self.labels, return_counts=True)
        print("Shape:", self.images.shape)
        print("Data type:", self.images.dtype)
        print("Label counts:", dict(zip(values.tolist(), counts.tolist())))


if __name__ == "__main__":
    b1 = ImageBatch(np.random.rand(3, 32, 32, 3), np.array([0, 1, 1]))
    b2 = ImageBatch(np.random.rand(2, 32, 32, 3), np.array([2, 0]))
    merged = b1 + b2
    merged.display_metadata()
    try:
        b1 + ImageBatch(np.random.rand(2, 16, 16, 3), np.array([0, 1]))
    except ValueError as e:
        print("Error:", e)