import logging
from pathlib import Path

import numpy as np
import pandas as pd

import wandb

logger = logging.getLogger("evaluate")


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def check_identical_prediction(predictions, min_std=0.1, min_distinct_ratio=0.1, capacity=134):

    total_errors = 0
    total_errors += (predictions.min(axis=1) == predictions.max(axis=1)).sum(axis=0)

    if (predictions.std(axis=1) < min_std).any():
        for i in range(predictions.shape[-1]):
            pred = predictions[:, :, i].reshape(-1)
            distinct_prediction = set(pred)
            distinct_ratio = len(distinct_prediction) / np.size(pred)

            if distinct_ratio < min_distinct_ratio:
                total_errors[i] += 1

    variation_ratio = total_errors / capacity

    if (variation_ratio > min_distinct_ratio).any():
        logger.error(
            bcolors.WARNING
            + "Validation error: Found turbines with (almost) identical predicted values!"
            + bcolors.ENDC
        )
        return -1

    return 0


def check_zero_prediction(predictions):
    if not (np.any(predictions, axis=1) == 0).sum() == 0:
        logger.error(bcolors.WARNING + "VALIDATION ERROR: Zero predicted values!" + bcolors.ENDC)
        return -1
    return 0


def mae_by_forecast(predictions, targets, valid_indices):
    aux = targets.copy()
    aux[~valid_indices] = 0
    aux = ~aux.any(axis=1)

    abs_error = np.abs(predictions - targets)
    abs_error[~valid_indices] = np.nan
    pred_maes = np.nanmean(abs_error, 1)
    pred_maes[aux] = 0
    pred_maes = pred_maes.sum(axis=0)

    return pred_maes


def rmse_by_forecast(predictions, targets, valid_indices):
    aux = targets.copy()
    aux[~valid_indices] = 0
    aux = ~aux.any(axis=1)

    square_error = np.power(predictions - targets, 2)
    square_error[~valid_indices] = np.nan
    pred_rmses = np.power(np.nanmean(square_error, 1), 0.5)
    pred_rmses[aux] = 0
    pred_rmses = pred_rmses.sum(axis=0)

    return pred_rmses


def general_mae(predictions, targets, valid_indices, first_reduce_idx=0, second_reduce_idx=1):
    abs_error = np.abs(predictions - targets)
    abs_error[~valid_indices] = np.nan
    pred_maes = np.nanmean(np.nanmean(abs_error, first_reduce_idx), second_reduce_idx)

    return pred_maes


def general_rmse(predictions, targets, valid_indices, first_reduce_idx=0, second_reduce_idx=1):
    square_error = np.power(predictions - targets, 2)
    square_error[~valid_indices] = np.nan
    pred_rmses = np.nanmean(
        np.power(np.nanmean(square_error, first_reduce_idx), 0.5), second_reduce_idx
    )

    return pred_rmses


def upload_numpy_array_to_wandb(exp, array, name):
    buffer = pd.DataFrame()
    buffer["idx"] = range(len(array))
    buffer[name] = array
    table = wandb.Table(dataframe=buffer)

    exp.log({name: table})


def log_errors_by_patv_value(exp, targets, preds, valid_indices):
    errors_df = pd.DataFrame(
        {
            "targets": targets.flatten() * 1000,
            "preds": preds.flatten() * 1000,
            "valid": valid_indices.flatten(),
        }
    )
    errors_df = errors_df[errors_df.valid == True].drop("valid", axis=1)

    errors_df["error"] = errors_df["preds"] - errors_df["targets"]
    errors_df["ae"] = np.abs(errors_df["error"])

    error_by_target = errors_df.groupby(
        pd.cut(errors_df.targets, np.linspace(0, 1500, 16))
    ).ae.mean()
    error_by_pred = errors_df.groupby(pd.cut(errors_df.preds, np.linspace(0, 1500, 16))).ae.mean()

    df_error_by_target = pd.DataFrame(error_by_target).reset_index().reset_index()
    df_error_by_pred = pd.DataFrame(error_by_pred).reset_index().reset_index()

    df_error_by_target["targets"] = df_error_by_target["targets"].astype("str")
    df_error_by_pred["preds"] = df_error_by_pred["preds"].astype("str")

    table_error_by_target = wandb.Table(dataframe=df_error_by_target)
    table_error_by_pred = wandb.Table(dataframe=df_error_by_pred)

    exp.log({"error_by_target": table_error_by_target})
    exp.log({"error_by_pred": table_error_by_pred})


def error_analysis(predictions, targets, valid_indices, settings):
    logger.info("\n\n\nSTARTING ERROR ANALYSIS...")

    exp = wandb.init(
        **settings["wandb"],
        name=settings["wandb_custom"]["run_name"],
        tags=settings["wandb_custom"]["user_name"],
        config=settings["envs"]
    )

    wandb.run.log_code(settings["wandb_custom"]["code_folder"])

    code_folder_path = Path(settings["wandb_custom"]["code_folder"])
    checkpoints_path = code_folder_path / "checkpoints"
    scalers_path = checkpoints_path / "scalers"

    if checkpoints_path.exists():
        wandb.save(
            str(checkpoints_path / "*"),
            base_path=str(code_folder_path),
            policy="end",
        )
        if scalers_path.exists():
            wandb.save(
                str(scalers_path / "*"),
                base_path=str(checkpoints_path),
                policy="end",
            )

    # Log error by patv value
    log_errors_by_patv_value(exp, targets, predictions, valid_indices)

    # Error vs predicted value
    np.abs(predictions.flatten() - targets.flatten())
    np.sqrt((predictions.flatten() - targets.flatten()) ** 2)

    # Analyze the error just like baidu does in their scoring script.
    mae_forecast = mae_by_forecast(predictions, targets, valid_indices)
    rmse_forecast = rmse_by_forecast(predictions, targets, valid_indices)
    scores = (mae_forecast + rmse_forecast) / 2

    # Analyze the error by forecasting horizon.
    mae_horizon = general_mae(
        predictions, targets, valid_indices, first_reduce_idx=0, second_reduce_idx=1
    )
    rmse_horizon = general_rmse(
        predictions, targets, valid_indices, first_reduce_idx=0, second_reduce_idx=1
    )

    # Analyze the error by turbID
    mae_turbID = general_mae(
        predictions, targets, valid_indices, first_reduce_idx=1, second_reduce_idx=1
    )
    rmse_turbID = general_rmse(
        predictions, targets, valid_indices, first_reduce_idx=1, second_reduce_idx=1
    )

    turb_scores = settings["envs"]["capacity"] * (mae_turbID + rmse_turbID) / 2

    upload_numpy_array_to_wandb(exp, mae_horizon, "MAE by horizon")
    upload_numpy_array_to_wandb(exp, rmse_horizon, "RMSE by horizon")
    upload_numpy_array_to_wandb(exp, turb_scores, "Scores by TurbID")

    exp.log(
        {
            "test_mae": mae_forecast.mean(),
            "test_mae_std": mae_forecast.std(),
            "test_rmse": rmse_forecast.mean(),
            "test_rmse_std": rmse_forecast.std(),
            "test_score": scores.mean(),
            "test_score_std": scores.std(),
        }
    )

    logger.info(
        "\n --- Final MAE: {}, RMSE: {} ---".format(mae_forecast.mean(), rmse_forecast.mean())
    )
    logger.info("--- Final Score --- \n\t{}".format(scores.mean()))

    # Run checks
    logger.error("\nRunning checks on the predicitons...")
    check_zero_prediction(predictions)
    check_identical_prediction(predictions)

    # Estimate model score
