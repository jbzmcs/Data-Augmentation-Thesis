import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import CONFIG, AugCase
from data_module import get_dataloaders
from model_module import LitResNet
from utils import log_results_to_csv, save_augmented_images


def run_experiment(case: AugCase, filter_classes=None):
    print(f"Running {case.name}...")

    CONFIG["case"] = case
    CONFIG["filter_classes"] = filter_classes  # Optional filtering

    # Load data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(case, CONFIG, filter_classes)
    print("Loaded classes:", class_names)

    # Save checkpoint with case + filter name
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{case.name}",
        filename="{epoch:02d}-{step}-" + "-".join(filter_classes or ["all"]),
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    # Model
    model = LitResNet(num_classes=len(class_names))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="auto",
        logger=False,
        callbacks=[checkpoint_callback]
    )

    # Train + Test
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    test_results = trainer.callback_metrics  # This includes test_acc and test_loss

    # Save results
    if CONFIG["save_augmented_images"]:
        sample_batch, _ = next(iter(train_loader))
        aug_batch = model.augment(sample_batch)
        save_augmented_images(sample_batch, aug_batch, tag=case.name)

    if CONFIG["log_csv"]:
        log_results_to_csv(model.metrics, case=case.name, config=CONFIG)

    print("Experiment complete!")
    print(f"Checkpoint saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case", type=int, required=True, choices=[1, 2, 3],
        help="Select augmentation case: 1 (train only), 2 (train+val), 3 (train+val+test)"
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=None,
        help="Optional: Filter specific classes (e.g. --filter noncrack thincrack). Default loads all classes."
    )

    args = parser.parse_args()
    selected_case = AugCase(args.case)
    run_experiment(selected_case, args.filter)
