"""
    This script divides the provided data into training, validation and testing splits.
    The user can specify the validation strategy (conventional or cross validation) with the cv_folds parameter
"""

import argparse
import shutil
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from tscv import GapKFold


def gap_finder(input_list: Union[List, np.array]):
    """
    This function divides a list of numbers into a generator of sublists made of consecutive numbers.
    For example [1,2,4,5] -> [1,2], [4,5]

    Args:
        input_list (Union[List, np.array]): A list of numbers
    """
    for k, g in groupby(enumerate(input_list), lambda i_x: i_x[0] - i_x[1]):
        yield list(map(itemgetter(1), g))


def interval_as_string(interval: Union[List, np.array]) -> str:
    starts_at = min(interval)
    ends_at = max(interval)
    return f"[{starts_at},{ends_at}]"


def preprocess_df(dataset_path: Path) -> pd.DataFrame:
    """
    Preprocess the raw dataframe before generating the splits.

    Args:
        dataset_path (Path): Dataset to preprocess

    Raises:
        ValueError: This error is raised if the incoming file is not a csv.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """

    if not str(dataset_path).endswith(".csv"):
        raise ValueError("The input dataset must be in csv format")

    df = pd.read_csv(dataset_path)

    # If a day does not contain a single valid observation, we will drop it.

    return df


def save_subfolds(
    df: pd.DataFrame, indices: Union[List, np.array], split: str, fold_path: Path
) -> None:
    """
    This function stores a dataframe split in several files. The need to use several files comes from the non continuity of the indices.
    For example, the specified indices may be [1,2,5,6] with a gap between 2 and 5.
    To avoid feeding the model with non continuous data, we store the values at the left and at the right of the gap ('subfolds') in a separate file.

    Args:
        df (pd.DataFrame): Input dataframe
        indices (Union[List, np.array]): The indices of a subslice of the days.unique() to be stored.
        split (str): Split (train, validation or test)
        fold_path (Path): Target folder where the data will be saved.
    """

    # Save each subfolt as a separate pickle file
    days = df.Day.unique()
    for index, subfold_indices in enumerate(gap_finder(indices), 1):
        print(
            f"->Subfold {index}:",
            interval_as_string(days[subfold_indices]),
        )

        subfold_path = fold_path / f"{split}" / f"data_{index}.pkl"

        # Create the folder structure if it doesn't exist already
        subfold_path.parent.mkdir(exist_ok=True, parents=True)

        # Save the subfold
        filtered_df = df[df.Day.isin(days[subfold_indices])]
        filtered_df.to_pickle(subfold_path)


def create_cv_folds(
    n: int, dataset_path: Path, target_folder_path: Path, n_test_days: int, gap: int = 0
) -> None:
    """
    Use the GapKFold strategy to create n cross validation folds.

    Args:
        n (int): Number of folds to generate.
        dataset_path (Path): Input dataset path
        target_folder_path (Path): Target path where the data will be stored.
        n_test_days (int): Number of days in the test split
        gap (int, optional): Number of gap days between the splits. Defaults to 0
    """

    cv = GapKFold(n_splits=n, gap_before=gap, gap_after=0)

    df = preprocess_df(dataset_path)
    days = df.Day.unique()

    if n_test_days > 0:
        trainval_df = df[df.Day.isin(days[: -n_test_days - gap])]
    else:
        trainval_df = df

    trainval_days = trainval_df.Day.unique()

    for split_index, (train_indices, val_indices) in enumerate(cv.split(trainval_days), 1):

        # Create a folder for the current cv fold
        fold_path = target_folder_path / f"fold-{split_index}"

        print(f"\nProcessing fold {split_index} of {n} consisting of the following days")

        print("Train split")
        save_subfolds(trainval_df, train_indices, "train", fold_path)
        print("Validation split")
        save_subfolds(trainval_df, val_indices, "validation", fold_path)

    if n_test_days > 0:
        print("\nTest split")
        test_path = target_folder_path / "test"
        test_df = df[df.Day.isin(days[-n_test_days:])]
        save_subfolds(test_df, range(len(test_df.Day.unique())), "train", test_path)


def create_train_test_splits(
    dataset_path: Path,
    target_folder_path: Path,
    n_test_days: int,
    n_validation_days: int,
    gap: int = 0,
) -> None:
    """
    Use the conventional train validation test strategy to create splits.

    Args:
        dataset_path (Path): Input dataset path
        target_folder_path (Path): Target path where the data will be stored.
        n_test_days (int): Number of days in the test split
        n_validation_days (int): Number of days in the validation split
        gap (int, optional): Number of gap days between the splits. Defaults to 0.
    """

    df = preprocess_df(dataset_path)

    train_indices = list(range(0, df.Day.max() - n_test_days - n_validation_days - 2 * gap))
    validation_indices = list(range(-n_test_days - gap - n_validation_days, -n_test_days - gap))
    test_indices = list(range(-n_test_days, -1))

    print("Train split")
    save_subfolds(df, train_indices, "train", target_folder_path)
    print("Validation split")
    save_subfolds(df, validation_indices, "validation", target_folder_path)
    print("Test split")
    save_subfolds(df, test_indices, "test", target_folder_path)


parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="Path to the wind power dataset", type=Path)
parser.add_argument(
    "target_folder_path",
    help="Path to the folder where the resulting folds will be saved",
    type=Path,
)
parser.add_argument(
    "n_folds",
    help="Desired number of folds, set this value to one if you only want a train test split",
    type=int,
    choices=list(range(1, 11)),
)
parser.add_argument(
    "--n_test_days",
    help="Number of days which will be used for the testing split",
    default=50,
    type=int,
)
parser.add_argument(
    "--n_val_days",
    help="Number of days which will be used for the validation split (if no cross validation)",
    default=50,
    type=int,
)
parser.add_argument(
    "--gap",
    help="Gap between the end of the training set and the beginning of the next set (in days)",
    default=0,
    type=int,
)
args = parser.parse_args()

if __name__ == "__main__":

    print("Creating datasets.")

    target_folder_path = args.target_folder_path

    # Remove the target folder contents if it already exists.
    if target_folder_path.is_dir():
        shutil.rmtree(target_folder_path)
    elif target_folder_path.is_file():
        raise NotADirectoryError("The provided path corresponds to a file, not to a directory.")

    if args.n_folds > 1:
        create_cv_folds(
            dataset_path=args.dataset_path,
            target_folder_path=args.target_folder_path,
            n_test_days=args.n_test_days,
            n=args.n_folds,
            gap=args.gap,
        )
    else:
        create_train_test_splits(
            dataset_path=args.dataset_path,
            target_folder_path=args.target_folder_path,
            n_test_days=args.n_test_days,
            n_validation_days=args.n_val_days,
            gap=args.gap,
        )
