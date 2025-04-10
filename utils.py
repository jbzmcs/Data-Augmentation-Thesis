# utils.py

import os
import csv
import torchvision.utils as vutils
from datetime import datetime
from config import CONFIG

def save_augmented_images(batch, batch_aug, tag):
    output_dir = os.path.join(CONFIG["output_path"], "augmented_images", tag)
    os.makedirs(output_dir, exist_ok=True)

    for i, (orig, aug) in enumerate(zip(batch, batch_aug)):
        vutils.save_image(orig, f"{output_dir}/orig_{i}.png")
        vutils.save_image(aug, f"{output_dir}/aug_{i}.png")

def log_results_to_csv(case_name, train_acc, val_acc):
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    log_file = os.path.join(CONFIG["output_path"], "results_log.csv")

    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Case", "Train Accuracy", "Val Accuracy"])
        writer.writerow([datetime.now(), case_name, f"{train_acc:.4f}", f"{val_acc:.4f}"])
