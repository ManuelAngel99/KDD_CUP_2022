from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Turn off the SettingWithCopyWarning
pd.set_option("mode.chained_assignment", None)


def test_slices_generator(
    input_df_path: Path, N: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        input_df_path (Path): Input data path
        N (int): Total number of slices to generte.
        seed (int): RNG seed

    Yields:
        Tuple[pd.DataFrame,pd.DataFrame]: input and target slices.
    """

    if not str(input_df_path).endswith(".pkl"):
        raise ValueError("The input dataframe must be a pickle")

    start = datetime(year=2022, month=1, day=1)
    df = pd.read_pickle(input_df_path)

    df["Hour"] = df.Tmstamp.apply(lambda x: int(x[0:2]))
    df["Min"] = df.Tmstamp.apply(lambda x: int(x[-2:]))
    df["Datetime"] = pd.to_datetime(
        start
        + pd.to_timedelta(df["Day"] - 1, "d")
        + pd.to_timedelta(df["Hour"], "h")
        + pd.to_timedelta(df["Min"], "m")
    )

    # Add a column that stores the information about an observation being abnormal.
    # This will be used during the evaluation to filter out the abnromal observations.

    nan_cond = pd.isna(df).any(axis=1)
    invalid_cond = (
        (df["Patv"] < 0)
        | ((df["Patv"] == 0) & (df["Wspd"] > 2.5))
        | ((df["Pab1"] > 89) | (df["Pab2"] > 89) | (df["Pab3"] > 89))
        | ((df["Wdir"] < -180) | (df["Wdir"] > 180) | (df["Ndir"] < -720) | (df["Ndir"] > 720))
    )

    df["Abnormal"] = (nan_cond) | invalid_cond

    input_length_in_days = 14
    predict_length_in_days = 2

    last_day = df.Day.max()

    # Sample N values from all the possible start times.
    start_datetimes = (
        df[df.Day <= last_day - input_length_in_days - predict_length_in_days][["Datetime"]]
        .drop_duplicates()
        .sample(n=N, replace=False, random_state=seed)
    )

    for idx, start_datetime in start_datetimes.iterrows():
        forecast_start_datetime = start_datetime + timedelta(days=input_length_in_days)
        input_end_datetime = forecast_start_datetime - timedelta(minutes=10)
        forecast_end_datetime = (
            forecast_start_datetime + timedelta(days=predict_length_in_days) - timedelta(minutes=10)
        )

        input_slice = df[
            (df.Datetime >= start_datetime[0]) & (df.Datetime <= input_end_datetime[0])
        ]

        forecast_slice = df[
            (df.Datetime >= forecast_start_datetime[0]) & (df.Datetime <= forecast_end_datetime[0])
        ]

        # Drop auxiliar columns
        input_slice = input_slice.drop(["Hour", "Min", "Datetime"], axis=1)
        forecast_slice = forecast_slice.drop(["Hour", "Min", "Datetime"], axis=1)

        yield input_slice, forecast_slice


class TestData(object):
    """
    Desc: Test Data
    """

    def __init__(
        self,
        data,
        task="MS",
        target="Patv",
        start_col=3,  # the start column index of the data one aims to utilize
        farm_capacity=134,
    ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.farm_capacity = farm_capacity

        self.df_raw = data
        self.df_data = deepcopy(self.df_raw)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)

        # Handling the missing values
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)
        self.numpy_df_data = self.df_data.to_numpy()

    def get_turbine(self, tid):
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == "MS":
            data = self.numpy_df_data[:, self.start_col :]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data[border1:border2]
        df = self.df_raw[border1:border2]
        return seq, df

    def get_all_turbines(self):
        seqs, dfs = [], []
        for i in range(self.farm_capacity):
            seq, df = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
        return seqs, dfs
