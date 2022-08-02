from pathlib import Path
from typing import List, Optional, Tuple

import metrics as m
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dataset import Scaler, WPFDataset
from torch import nn
from torch.utils import data


class PLWrapper(pl.LightningModule):
    def __init__(
        self,
        settings: dict,
        model: nn.Module,
        turbines: List,
        train_df_path: Path,
        val_df_path: Path,
        checkpoints_path: Optional[Path] = None,
    ):
        super().__init__()
        self.settings = settings
        self.model = model
        self.turbines = turbines
        self.train_df_path = train_df_path
        self.val_df_path = val_df_path

        self.save_hyperparameters(ignore=["model"])

        self.best_loss = 1e30

        self.scaler = Scaler()
        self.tgt_scaler = Scaler()

        self.checkpoints_path = checkpoints_path

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.settings["lr"], weight_decay=self.settings["weight_decay"]
        )

    def inference_sample_and_log_metrics(self, batch: Tuple, log_pre_string: str) -> float:
        X, y, abnormal = batch

        X_scaled = self.scaler.transform(X)
        y_scaled = self.tgt_scaler.transform(y)

        out = self.model(X_scaled)

   

        out_descaled = self.tgt_scaler.inverse_transform(out)

        # Calculate the model's loss function
        mse_loss = F.mse_loss(out, y_scaled)
        mae_loss = F.l1_loss(out_descaled/1000, y/1000)
        loss =  mse_loss

        blp = m.baidu_loss_proxy(out_descaled / 1000, y / 1000, ~abnormal)
        mae_loss, rmse_loss = m.filtered_scores(out_descaled / 1000, y / 1000, ~abnormal)

        # W&B Logging
        if not blp.isnan().any():
            self.log(log_pre_string + "_blp", blp)
        if not mae_loss.isnan().any():
            self.log(log_pre_string + "_mae", mae_loss)
        if not rmse_loss.isnan().any():
            self.log(log_pre_string + "_rmse", rmse_loss)

        self.log(log_pre_string + "_mse_loss_dirty", mse_loss)
        self.log(log_pre_string + "_mae_loss_dirty", mae_loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.inference_sample_and_log_metrics(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.inference_sample_and_log_metrics(batch, "val")
        return loss

    def validation_epoch_end(self, val_outs):
        avg_score = torch.mean(torch.stack(val_outs))
        self.best_loss = min(avg_score, self.best_loss)
        self.log("best_val_score", self.best_loss)

    def train_dataloader(self):
        loader = self.create_dataloader(
            df_path=self.train_df_path,
            batch_size=self.settings["train_batch_size"],
            shuffle=True,
            drop_last=True,
            get_scaler=True,
        )

        return loader

    def val_dataloader(self):
        loader = self.create_dataloader(
            df_path=self.val_df_path,
            batch_size=self.settings["val_batch_size"],
            shuffle=False,
            drop_last=False,
        )

        return loader

    def create_dataloader(self, df_path, batch_size, shuffle, drop_last=False, get_scaler=False):

        dataset = WPFDataset(
            path=df_path,
            input_window_length=self.settings["input_length"],
            variables=self.settings["inputs"],
            targets=self.settings["targets"],
            turbines=self.turbines,
        )

        aux_run_name = self.settings["run_name"]
        if get_scaler:
            self.scaler, self.tgt_scaler = dataset.create_scaler(device="cuda")
            self.scaler.save(self.checkpoints_path / "scalers", f"{aux_run_name}_ins")
            self.tgt_scaler.save(self.checkpoints_path / "scalers", f"{aux_run_name}_tgt")

        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4,
            drop_last=drop_last,
        )

        return dataloader
