# Scenario: In professional AI workflows, data is rarely handled as raw arrays. You need to create a robust ImageBatch container that manages both image data (4D NumPy array) and their corresponding labels (1D NumPy array), while ensuring all operations are logged for experiment tracking.
# Requirements:
# Your class must inherit from BaseBatch class given below and implement necessary functions.
# Create a LoggingMixin that provides a method log_operation(message). This should print a timestamped message whenever data is modified.
# The ImageBatch Class:
# Initialization: The constructor should accept a NumPy array of shape (N,H,W,C) and a label array of shape (N,) where  N = number of samples, H = image height, W = image width and C = image channels.
# Dunder Addition: Implement __add__. When two ImageBatch objects are added (batch1 + batch2), the result should be a new ImageBatch object containing the concatenated images and labels.
# Array Inspection: Implement display_metadata() to print the shape, data type, and the unique counts of labels using np.unique().
# NumPy Operations: Within the addition logic, ensure the labels remain aligned with their respective images.
# Defense: In the comments you must defend your approach for addition of 2 batch objects. 
# Technical Constraints:
# Ensure the __add__ method logs the operation: "Concatenating two batches of size X and Y."
# Validate that the number of images matches the number of labels during initialization.
# During addition if image dimensions (Height, Width, Channels) don't match between two batches by raise a ValueError.



from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

class BaseBatch(ABC):
    @abstractmethod
    def __len__(self):

    @abstractmethod
    def __add__(self, other): 

    @abstractmethod
    def display_metadata(self):


class LoggingMixin:
    """Provides timestamped logging for experiment tracking."""

    def log_operation(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")


class ImageBatch(LoggingMixin, BaseBatch):
    def __init__(self, images, labels):
        images = np.asarray(images)
        labels = np.asarray(labels)

        if images.ndim !=