import os
import csv
import torchvision.utils as vutils
from datetime import datetime
from config import CONFIG


def save_augmented_images(orig_batch, aug_batch, tag, class_names, labels):
    output_dir = os.path.join(CONFIG["output_path"], "augmented_images", tag)
    os.makedirs(output_dir, exist_ok=True)

    for i, (orig, aug, label) in enumerate(zip(orig_batch, aug_batch, labels)):
        class_name = class_names[label.item()]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        vutils.save_image(orig, os.path.join(class_dir, f"orig_{i}.png"))
        vutils.save_image(aug, os.path.join(class_dir, f"aug_{i}.png"))


def log_results_to_csv(metrics, case, model_name, config, class_names):
    os.makedirs(config["output_path"], exist_ok=True)
    log_file = os.path.join(config["output_path"], "results_log.csv")

    write_header = not os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [
                "timestamp", "case", "model", "train_acc", "val_acc", "test_acc",
                "train_precision", "val_precision", "test_precision",
                "train_recall", "val_recall", "test_recall",
                "train_f1", "val_f1", "test_f1"
            ]
            # Add per-class metrics for each class
            for class_name in class_names:
                header.extend([
                    f"test_precision_{class_name}",
                    f"test_recall_{class_name}",
                    f"test_f1_{class_name}"
                ])
            writer.writerow(header)

        row = [
            datetime.now().isoformat(),
            case,
            model_name,
            metrics.get("train_acc", 0.0),
            metrics.get("val_acc", 0.0),
            metrics.get("test_acc", 0.0),
            metrics.get("train_precision", 0.0),
            metrics.get("val_precision", 0.0),
            metrics.get("test_precision", 0.0),
            metrics.get("train_recall", 0.0),
            metrics.get("val_recall", 0.0),
            metrics.get("test_recall", 0.0),
            metrics.get("train_f1", 0.0),
            metrics.get("val_f1", 0.0),
            metrics.get("test_f1", 0.0)
        ]
        # Add per-class metrics
        for class_name in class_names:
            row.extend([
                metrics.get(f"test_precision_class_{class_names.index(class_name)}", 0.0),
                metrics.get(f"test_recall_class_{class_names.index(class_name)}", 0.0),
                metrics.get(f"test_f1_class_{class_names.index(class_name)}", 0.0)
            ])
        writer.writerow(row)