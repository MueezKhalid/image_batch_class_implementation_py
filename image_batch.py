from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

class BaseBatch(ABC):
    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __add__(self, other): ...

    @abstractmethod
    def display_metadata(self): ...


class LoggingMixin:
    """Provides timestamped logging for experiment tracking."""

    def log_operation(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")


class ImageBatch(LoggingMixin, BaseBatch):
    def __init__(self, images, labels):
        images = np.asarray(images)
        labels = np.asarray(labels)

        if images.ndim != 4:
            raise ValueError(f"images must be 4D (N,H,W,C), got {images.ndim}D")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D (N,), got {labels.ndim}D")

        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of images ({images.shape[0]}) does not match "
                f"number of labels ({labels.shape[0]})"
            )

        self.images = images
        self.labels = labels
        self.log_operation(f"Created ImageBatch of size {len(self)}.")

    def __len__(self):
        return self.images.shape[0]

    def __add__(self, other):
        if not isinstance(other, ImageBatch):
            return NotImplemented

        if self.images.shape[1:] != other.images.shape[1:]:
            raise ValueError(
                f"Image dimensions (H, W, C) mismatch: "
                f"{self.images.shape[1:]} vs {other.images.shape[1:]}"
            )

        self.log_operation(
            f"Concatenating two batches of size {len(self)} and {len(other)}."
        )
        new_images = np.concatenate((self.images, other.images), axis=0)
        new_labels = np.concatenate((self.labels, other.labels), axis=0)
        return ImageBatch(new_images, new_labels)

    def display_metadata(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Shape: {self.images.shape}")
        print(f"Data type: {self.images.dtype}")
        print(f"Label counts: {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    b1 = ImageBatch(np.random.rand(3, 32, 32, 3), np.array([0, 1, 1]))
    b2 = ImageBatch(np.random.rand(2, 32, 32, 3), np.array([2, 0]))
    merged = b1 + b2
    merged.display_metadata()

    try:
        b1 + ImageBatch(np.random.rand(2, 16, 16, 3), np.array([0, 1]))
    except ValueError as e:
        print("Caught expected error:", e)