import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils import data


class Scaler:
    def __init__(self, device="cuda"):
        self.mean = torch.zeros((1), device=device)
        self.std = torch.ones((1), device=device)
        self.device = device

    def fit(self, df):
        self.mean = torch.from_numpy(df.mean().to_numpy()).to(
            device=torch.device(self.device), dtype=torch.float32
        )
        self.std = torch.from_numpy(df.std().to_numpy()).to(
            device=torch.device(self.device), dtype=torch.float32
        )
        assert (self.std > 1e-3).all()

        return self

    def transform(self, data):
        transformed = (data - self.mean) / self.std
        return torch.nan_to_num(transformed, nan=0)

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def to(self, device):
        self.mean.to(device)
        self.std.to(device)

    def save(self, path, name):
        base_path = Path(path)
        base_path.mkdir(exist_ok=True, parents=True)

        torch.save(self.mean, base_path / f"mean_{name}.pt")
        torch.save(self.std, base_path / f"std_{name}.pt")

    @classmethod
    def load(cls, path, name, device="cuda"):
        base_path = Path(path)

        instance = cls()
        instance.mean = torch.load(base_path / f"mean_{name}.pt").to(device)
        instance.std = torch.load(base_path / f"std_{name}.pt").to(device)

        return instance


class WPFDataset(data.Dataset):
    def __init__(
        self,
        path: Path,
        input_window_length: int,
        variables: List[str],
        targets: List[str],
        turbines: List[int] = list(range(1, 135, 1)),
        stride: int = 1,
    ):

        self.raw_dfs = list()

        if type(path) == type(pd.DataFrame()):
            self.raw_dfs = [path]
        else:
            for file in path.iterdir():
                self.raw_dfs.append(pd.read_pickle(file))

        self.turbines = sorted(turbines)

        self.variables = variables
        self.targets = targets
        self.stride = stride
        self.input_window_length = input_window_length

        self.recalculate_slices(
            input_window_length=input_window_length,
            variables=variables,
            targets=targets,
            turbines=turbines,
            stride=stride,
        )

    def data_preprocess_fn(self, dfs: List[pd.DataFrame], turbines: List) -> List[pd.DataFrame]:
        processed_dfs = list()

        for df in dfs:
            df = df[df["TurbID"].isin(turbines)]
            nan_cond = pd.isna(df).any(axis=1)
            invalid_cond = (
                (df["Patv"] < 0)
                | ((df["Patv"] == 0) & (df["Wspd"] > 2.5))
                | ((df["Pab1"] > 89) | (df["Pab2"] > 89) | (df["Pab3"] > 89))
                | (
                    (df["Wdir"] < -180)
                    | (df["Wdir"] > 180)
                    | (df["Ndir"] < -720)
                    | (df["Ndir"] > 720)
                )
            )
            df["Abnormal"] = nan_cond | invalid_cond
            processed_dfs.append(df)

        return processed_dfs

    def create_scaler(self, device):
        scaler_x = Scaler(device=device)
        scaler_y = Scaler(device=device)
        scaler_x.fit(self.concat_dfs[self.variables])
        scaler_y.fit(self.concat_dfs[self.targets])

        return scaler_x, scaler_y

    def recalculate_slices(
        self,
        input_window_length: int,
        variables: List[str],
        targets: List[str],
        turbines: List[int] = list(range(1, 135, 1)),
        stride: int = 1,
    ) -> None:

        self.variables = variables
        self.targets = targets
        self.stride = stride
        self.input_window_length = input_window_length
        self.turbines = sorted(turbines)

        # Filter the df & preprocess it
        self.dfs = self.data_preprocess_fn(self.raw_dfs, turbines)
        self.concat_dfs = pd.concat(self.dfs)

        self.n_turbines = len(turbines)
        self.output_window_length = 60
        self.total_sample_length = self.input_window_length + self.output_window_length
        self.rows_per_turbine = [len(df) / self.n_turbines for df in self.dfs]
        self.lengths = [
            (1 + math.floor((rows - self.total_sample_length) / self.stride)) * self.n_turbines
            for rows in self.rows_per_turbine
        ]

        self.target_slices = [
            torch.tensor(slice[self.targets].to_numpy(), dtype=torch.float32)
            for idx, slice in self.concat_dfs.groupby("TurbID")
        ]

        self.data_slices = [
            torch.tensor(slice[self.variables].to_numpy(), dtype=torch.float32)
            for idx, slice in self.concat_dfs.groupby("TurbID")
        ]

        self.abnormal_slices = [
            torch.tensor(slice["Abnormal"].to_numpy(), dtype=torch.bool)
            for idx, slice in self.concat_dfs.groupby("TurbID")
        ]

    def __len__(self) -> int:
        return sum(self.lengths)

    def __repr__(self):
        return self.concat_dfs.__repr__()

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        slice_idx = math.floor(self.n_turbines * idx / (len(self)))
        TurbID = self.turbines[slice_idx]

        pos = (self.stride * idx) % (len(self) / self.n_turbines)

        if pos >= (self.lengths[0]) / self.n_turbines:
            pos = pos + self.total_sample_length - 1

        TurbID = int(TurbID)
        pos = int(pos)

        inputs = self.data_slices[slice_idx][pos : pos + self.input_window_length]
        outputs = self.target_slices[slice_idx][
            pos + self.input_window_length : pos + self.total_sample_length
        ]

        abnormals = self.abnormal_slices[slice_idx][
            pos + self.input_window_length : pos + self.total_sample_length
        ][:]

        assert inputs.shape == torch.Size([self.input_window_length, len(self.variables)])
        assert outputs.shape == torch.Size(
            [self.output_window_length, len(self.targets)]
        ), f"wrong shape for outputs: {outputs.shape}"
        assert abnormals.shape == torch.Size([self.output_window_length])

        return (
            inputs,
            outputs,
            abnormals,
        )
