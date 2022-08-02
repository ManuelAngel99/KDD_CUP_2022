import argparse
import shutil
from pathlib import Path

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import seed

from model import BaselineGruModel
from pl_wrapper import PLWrapper
from prepare import prep_env

parser = argparse.ArgumentParser()
parser.add_argument(
    "train_data_path",
    help="Path to the folder containing the train dataset",
    type=Path,
)
parser.add_argument(
    "val_data_path",
    help="Path to the folder containing the validation dataset",
    type=Path,
)

parser.add_argument(
    "wb_project",
    help="W&B project name",
    type=str,
)

parser.add_argument(
    "--seed",
    help="Random seed",
    type=int,
)
parser.add_argument("--run_name", help="WANDB run name", type=str, default="train")

args = parser.parse_args()
if __name__ == "__main__":
    # Remove any previous checkpoints if present
    script_folder_path = Path(__file__).parent
    checkpoints_path = script_folder_path / f"checkpoints_{args.run_name}"
    if checkpoints_path.exists():
        shutil.rmtree(str(checkpoints_path))

    rng_seed = args.seed or 2022

    print(checkpoints_path)
    print(f"Starting training")
    wandb.finish()

    # W&B Logging
    wandb_logger = WandbLogger(project=args.wb_project, log_model="all", name=args.run_name)

    wandb_logger._experiment.log_code(
        ".", include_fn=lambda path: (path.endswith(".py") or path.endswith(".ipynb"))
    )

    # Load the model settings
    settings = prep_env()

    # Initialize random seeds.
    seed.seed_everything(rng_seed, True)

    # Get our model
    model = BaselineGruModel(settings, cuda=True)

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_blp",
        mode="min",
        dirpath=str(checkpoints_path),
        filename=f"model_{args.run_name}",
    )

    settings["run_name"] = args.run_name

    trainer = pl.Trainer(
        default_root_dir=str(checkpoints_path),
        max_epochs=settings["train_epochs"],
        accelerator="gpu",
        auto_lr_find=False,
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
        val_check_interval=0.1,
        # gradient_clip_val=None,
        accumulate_grad_batches=1,
    )

    model_wrapper = PLWrapper(
        settings=settings,
        model=model,
        turbines=list(range(1, 135)),
        train_df_path=args.train_data_path,
        val_df_path=args.val_data_path,
        checkpoints_path=checkpoints_path,
    )

    trainer.fit(model_wrapper)
