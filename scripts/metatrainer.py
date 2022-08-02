import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "training_script",
    help="Path to the model training script",
    type=Path,
)

parser.add_argument(
    "cv_folders_path",
    help="Path containing the folders of the cv datasets",
    default=Path,
)
parser.add_argument(
    "wb_project",
    help="W&B project name",
    type=str,
)

parser.add_argument(
    "prefix_name_run",
    help="Prefix name run",
    type=str,
)
args = parser.parse_args()
if __name__ == "__main__":
    print("---- MODEL METATRAINER -----\n")

    for cv_folder in Path(args.cv_folders_path).iterdir():
        print("\n\n")
        print("BEGINNING TRAINING FOR ", cv_folder)
        os.system(
            f"poetry run python {args.training_script} {cv_folder/'train'} {cv_folder/'validation'} {args.wb_project} --run_name {cv_folder.name}"
        )


# poetry run python ./archive/$$folder/training.py /code/data/train_val_test/train/ /code/data/train_val_test/test/ $$wandb_project
