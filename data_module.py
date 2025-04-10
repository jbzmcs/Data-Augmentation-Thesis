import os
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import CONFIG


def filter_dataset_by_classes(dataset, classes_to_keep):
    """
    Filters an ImageFolder dataset to only include specified classes.
    Adjusts the samples and targets accordingly.
    """
    # Determine the indices of the samples that belong to the desired classes.
    filtered_samples = []
    filtered_targets = []
    # Get the mapping from class names to indices
    original_class_to_idx = dataset.class_to_idx

    # Create a set of class indices to keep.
    indices_to_keep = {original_class_to_idx[c] for c in classes_to_keep if c in original_class_to_idx}

    for sample, target in zip(dataset.samples, dataset.targets):
        if target in indices_to_keep:
            filtered_samples.append(sample)
            filtered_targets.append(target)

    # Update the dataset.
    dataset.samples = filtered_samples
    dataset.targets = filtered_targets
    # Optionally, update dataset.classes and class_to_idx for clarity.
    # For a proper update, re-map the indices for only the filtered classes.
    new_classes = sorted(list(classes_to_keep))
    new_class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)}

    # Map old target indices to new ones.
    new_targets = []
    new_samples = []
    for sample, target in zip(filtered_samples, filtered_targets):
        # Find class name corresponding to the original index.
        for cls_name, orig_idx in original_class_to_idx.items():
            if orig_idx == target and cls_name in new_class_to_idx:
                new_targets.append(new_class_to_idx[cls_name])
                new_samples.append(sample)
                break

    dataset.samples = new_samples
    dataset.targets = new_targets
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx

    return dataset


def get_dataloaders(case, config, filter_classes=None):
    """
    Loads the train, validation, and test datasets from subfolders.
    Optionally filters to only the provided classes (e.g., ['noncrack', 'thincrack']).
    """
    dataset_path = config["dataset_path"]

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")

    # Base transform (without augmentation) for all splits.
    base_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets using ImageFolder.
    train_dataset = ImageFolder(train_path, transform=base_transform)
    val_dataset = ImageFolder(val_path, transform=base_transform)
    test_dataset = ImageFolder(test_path, transform=base_transform)

    # Optionally filter datasets to only include select classes.
    if filter_classes is not None:
        train_dataset = filter_dataset_by_classes(train_dataset, filter_classes)
        val_dataset = filter_dataset_by_classes(val_dataset, filter_classes)
        test_dataset = filter_dataset_by_classes(test_dataset, filter_classes)

    # Create data loaders.
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"])

    # Return the loaders and the class names (updated, if filtered).
    return train_loader, val_loader, test_loader, train_dataset.classes

# Example usage:
# For early experiments, you might call:
# train_loader, val_loader, test_loader, class_names = get_dataloaders(case, CONFIG, filter_classes=["noncrack", "thincrack"])
# Later, you can call get_dataloaders(case, CONFIG) to use all classes.

class PanelDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32, num_workers=4, filter_classes=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter_classes = filter_classes
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None

    def setup(self, stage=None):
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            case=CONFIG["case"],
            config=CONFIG,
            filter_classes=self.filter_classes
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    @property
    def num_classes(self):
        return len(self.class_names)