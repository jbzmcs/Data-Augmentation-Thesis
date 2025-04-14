import os
import csv
import torchvision.utils as vutils
from datetime import datetime
from config import CONFIG


def save_augmented_images(orig_batch, aug_batch, tag):
    output_dir = os.path.join(CONFIG["output_path"], "augmented_images", tag)
    os.makedirs(output_dir, exist_ok=True)

    for i, (orig, aug) in enumerate(zip(orig_batch, aug_batch)):
        vutils.save_image(orig, os.path.join(output_dir, f"orig_{i}.png"))
        vutils.save_image(aug, os.path.join(output_dir, f"aug_{i}.png"))

# Stores all results, logs train, val, and test accuracy
def log_results_to_csv(metrics, case, config):
    os.makedirs(config["output_path"], exist_ok=True)
    log_file = os.path.join(config["output_path"], "results_log.csv")

    write_header = not os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "case", "train_acc", "val_acc", "test_acc"])
        writer.writerow([
            datetime.now().isoformat(),
            case,
            metrics.get("train_acc"),
            metrics.get("val_acc"),
            metrics.get("test_acc")
        ])

