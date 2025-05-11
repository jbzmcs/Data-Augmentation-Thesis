import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import CONFIG, AugCase
from data_module import get_dataloaders
from model_module import LitModel
from utils import log_results_to_csv, save_augmented_images


def run_experiment(case: AugCase, filter_classes=None, model_name="resnet18"):
    print(f"Running {case.name} with model {model_name}...")

    CONFIG["case"] = case
    CONFIG["filter_classes"] = filter_classes  # Optional filtering

    # Load data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(case, CONFIG, filter_classes, model_name)
    print("Loaded classes:", class_names)

    # Save checkpoint with case + filter + model name
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{case.name}/{model_name}",
        filename="{epoch:02d}-{step}-" + "-".join(filter_classes or ["all"]),
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    # Model
    model = LitModel(model_name=model_name, num_classes=len(class_names), case=case)

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
        sample_batch, labels = next(iter(train_loader))
        aug_batch = model.augment(sample_batch)
        save_augmented_images(sample_batch, aug_batch, tag=f"{case.name}_{model_name}", class_names=class_names, labels=labels)

    if CONFIG["log_csv"]:
        log_results_to_csv(model.metrics, case=case.name, model_name=model_name, config=CONFIG, class_names=class_names)

    print("Experiment complete!")
    print(f"Checkpoint saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", type=int, choices=[1, 2, 3],
        help="Select augmentation case: 1 (train only), 2 (train+val), 3 (train+val+test)"
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=None,
        help="Optional: Filter specific classes (e.g. --filter noncrack thincrack). Default loads all classes."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "alexnet", "vgg16", "inception_v3"],
        help="Select model: resnet18, alexnet, vgg16, or inception_v3"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 12 experiments (3 cases x 4 models)"
    )

    args = parser.parse_args()

    if args.all:
        cases = [AugCase.CASE1, AugCase.CASE2, AugCase.CASE3]
        models = ["resnet18", "alexnet", "vgg16", "inception_v3"]
        for case in cases:
            for model in models:
                run_experiment(case, args.filter, model)
    elif args.case and args.model:
        selected_case = AugCase(args.case)
        run_experiment(selected_case, args.filter, args.model)
    else:
        parser.error("Must specify either --all or both --case and --model")