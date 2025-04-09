# config.py
from enum import Enum

class AugCase(Enum):
    CASE1 = 1
    CASE2 = 2
    CASE3 = 3

# Configurable parameters (everything but the case)
CONFIG = {
    "dataset_path": "./Panel Classifier Dataset",
    "output_path": "./outputs",
    "save_augmented_images": True,
    "log_csv": True,
    "batch_size": 32,
    "num_workers": 4,
    "max_epochs": 10
}
