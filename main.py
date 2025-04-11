
import argparse
import pytorch_lightning as pl
from config import CONFIG, AugCase
from data_module import get_dataloaders
from model_module import LitResNet
from utils import log_results_to_csv, save_augmented_images


def run_experiment(case: AugCase):
    print(f"Running {case.name}...")

    # Load data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(case, CONFIG)
    print("Loaded classes:",class_names)
    # Set the case in global config
    CONFIG["case"] = case

    # Model
    model = LitResNet(num_classes=len(class_names)) # Fix: removed extra args to accept CONFIG["case"] in CLI

    # Training & Test
    trainer = pl.Trainer(max_epochs=CONFIG["max_epochs"], accelerator="gpu", logger=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Save results
    if CONFIG["save_augmented_images"]:
        save_augmented_images(train_loader, model, case=case.name, config=CONFIG)

    if CONFIG["log_csv"]:
        log_results_to_csv(model.metrics, case=case.name, config=CONFIG)

    print("Experiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    selected_case = AugCase(args.case)
    run_experiment(selected_case)
