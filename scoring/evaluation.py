import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from types import ModuleType
from typing import Tuple

import metrics
import numpy as np
import pandas as pd
import yaml
from error_analysis import error_analysis
from loader import Loader

from scoring.test_data import TestData, test_slices_generator

logger = logging.getLogger("evaluate")


class EvaluationError(Exception):
    """
    Desc:
        Customize the Exception for Evaluation
    """

    def __init__(self, err_message: str):
        Exception.__init__(self, err_message)


def performance(
    envs: dict,
    idx: int,
    prediction: np.array,
    ground_truth: np.array,
    ground_truth_df: pd.DataFrame,
) -> Tuple[float, float, float]:
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        envs: Dictionary of parameters supplied to the model
        idx: Current prediction index
        prediction: Model forecasts
        ground_truth: Ground truths
        ground_truth_df: Ground truths dataframe
    Returns:
        MAE, RMSE and Accuracy
    """
    (
        overall_mae,
        overall_rmse,
        _,
        overall_latest_rmse,
        all_valid_indices,
    ) = metrics.regressor_detailed_scores(prediction, ground_truth, ground_truth_df, envs)
    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    if overall_latest_rmse < 0:
        raise EvaluationError(
            "The RMSE of the last 24 hours is negative ({}) in the {}-th prediction"
            "".format(overall_latest_rmse, idx)
        )
    acc = 1 - overall_latest_rmse / envs["capacity"]
    return overall_mae, overall_rmse, acc, all_valid_indices


def predict_and_test(
    envs: dict,
    data: pd.DataFrame,
    idx: int,
    forecast_module: ModuleType,
    flag: str = "predict",
):
    """
    Desc:
        Do the prediction or get the ground truths
    Args:
        envs: Dictionary of parameters supplied to the model
        data: Dataframe
        idx: Current prediction index
        forecast_module: The model's forecasting module
        data: Dataframe
        flag: Either "predict" or "test". Select the task to perform
    Returns:
        A result dict containng the predictions and the ground truths dataframe if flag is set to test
    """

    if "predict" == flag:
        # Save the dataset into a temp file to emulate baidu's script.
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #    path_to_test_file = Path(tmp_dir) / f"{idx:08d}.pkl"
        #    data.to_pickle(path_to_test_file)

        envs["path_to_test_x"] = Path(__file__) / f"{idx:08d}.pkl"
        envs["custom_evaluation_dataframe"] = data
        prediction = forecast_module.forecast(envs)

        return {"prediction": prediction}

    elif flag == "test":
        test_data = TestData(data=data, start_col=envs["start_col"])
        turbines, raw_turbines = test_data.get_all_turbines()
        test_ys = []
        for turbine in turbines:
            test_ys.append(turbine[: envs["output_len"], -envs["out_var"] - 1 : -1])
        return {
            "ground_truth_y": np.array(test_ys),
            "ground_truth_df": raw_turbines,
        }
    else:
        raise EvaluationError(
            "Unsupported evaluation task (only 'predict' or 'test' is acceptable)! "
        )


def setup_environment(path_to_src_dir: Path, settings: dict) -> dict:
    """
    Desc:
        Load the model's prediction module and create the envs dictionary containing the model's parameters.
    Args:
        path_to_src_dir (Path): Path to the folder containing the model.
        settings (dict): Dictionary containg the script settings.

    Returns:
        dict: A dictionary containing the parameter's available to the model during prediction.
    """

    # Set up the initial environment
    path_to_prep_script = os.path.join(path_to_src_dir, "prepare.py")
    if not os.path.exists(path_to_prep_script):
        raise EvaluationError("The preparation script, i.e. 'prepare.py', does NOT exist! ")
    prep_module = Loader.load(path_to_prep_script)
    envs = prep_module.prep_env()

    for req_key in settings["script_settings"]["REQUIRED_ENV_VARS"]:
        if req_key not in envs:
            raise EvaluationError(
                "Key error: '{}'. The variable {} "
                "is missing in the prepared experimental settings! ".format(req_key, req_key)
            )

    envs.update(settings["envs"])
    if "is_debug" not in envs:
        envs["is_debug"] = False

    if envs["framework"] not in settings["script_settings"]["SUPPORTED_FRAMEWORKS"]:
        raise EvaluationError(
            "Unsupported machine learning framework: {}. "
            "The supported frameworks are 'base', 'paddlepaddle', 'pytorch', "
            "and 'tensorflow'".format(envs["framework"])
        )

    envs["pred_file"] = os.path.join(path_to_src_dir, envs["pred_file"])
    envs["checkpoints"] = os.path.join(path_to_src_dir, envs["checkpoints"])

    return envs


def evaluate(
    path_to_test_df: Path,
    path_to_src_dir: Path,
    n_runs: int,
    settings: dict,
    seed: int = 1234,
    analyze_errors: bool = False,
    save_results: bool = False,
) -> dict:
    """
    The evaluation function creates n_runs test samples from the dataframe located at path_to_test_df and evaluates the model performance.


    Args:
        path_to_test_df (Path): Path of the test split dataframe.
        path_to_src_dir (Path): Path of the model to evaluate.
        n_runs (int): Number of different test samples to use._
        settings (dict): Dictionary containg the script settings.

    Returns:
        dict: A dictionary containing the model scores.
    """

    begin_time = time.time()
    start_test_time = begin_time

    # Setup the environment for the evaluation.
    envs = setup_environment(path_to_src_dir, settings)
    settings["envs"].update(envs)

    maes, rmses, accuracies = [], [], []
    forecast_module = Loader.load(envs["pred_file"])

    start_forecast_time = start_test_time
    end_forecast_time = start_forecast_time

    score_dict = dict()

    predictions, targets, valid_indices = [], [], []

    for i, (test_x, test_y) in enumerate(
        test_slices_generator(path_to_test_df, N=n_runs, seed=seed)
    ):
        start_forecast_time = time.time()

        # Get model forecasts
        pred_res = predict_and_test(envs, test_x, i, forecast_module, flag="predict")
        prediction = pred_res["prediction"]

        # Get ground truths
        gt_res = predict_and_test(envs, test_y, i, forecast_module, flag="test")
        gt_ys = gt_res["ground_truth_y"]
        gt_turbines = gt_res["ground_truth_df"]

        # Calculate the model performance on this run.
        tmp_mae, tmp_rmse, tmp_acc, val_idx = performance(envs, i, prediction, gt_ys, gt_turbines)

        end_forecast_time = time.time()
        forecasting_latency = end_forecast_time - start_forecast_time

        logger.warning(
            "\n\tThe {}-th prediction -- "
            "RMSE: {:.5f}, MAE: {:.5f}, Score: {:.5f}, "
            "and Accuracy: {:.4f}% took {:.4f} seconds".format(
                i,
                tmp_rmse,
                tmp_mae,
                (tmp_rmse + tmp_mae) / 2,
                tmp_acc * 100,
                forecasting_latency,
            )
        )

        # Accuracy is lower than Zero, which means that the RMSE of this prediction is too large,
        # which also indicates that the performance is probably poor and not robust
        if tmp_acc <= 0 and not envs["is_debug"]:

            raise EvaluationError(
                "Accuracy ({:.3f}) is lower than Zero, which means that "
                "the RMSE (in latest 24 hours) of the {}th prediction "
                "is too large!".format(tmp_acc, i)
            )

        maes.append(tmp_mae)
        rmses.append(tmp_rmse)
        accuracies.append(tmp_acc)

        predictions.append(prediction)
        targets.append(gt_ys)

        valid_indices.append(val_idx)

    avg_mae = np.array(maes).mean()
    avg_rmse = np.array(rmses).mean()
    total_score = (avg_mae + avg_rmse) / 2

    score_dict["avg_mae"] = avg_mae
    score_dict["avg_rmse"] = avg_rmse
    score_dict["total_score"] = total_score

    print("\n --- Final MAE: {}, RMSE: {} ---".format(avg_mae, avg_rmse))
    print("--- Final Score --- \n\t{}".format(total_score))
    print("Calculating the scores took {:.5f} secs".format(time.time() - start_test_time))

    

    if save_results:
        save_path = Path(settings["args"]["model_path"]) / "scoring_outputs"
        print(f"SAVING PREDICTIONS to {str(save_path)}... \n")

        if save_path.exists():
            shutil.rmtree(str(save_path))

        save_path.mkdir(exist_ok=True, parents=True)

        np.save(str(save_path / "preds"), predictions)
        np.save(str(save_path / "targets"), targets)
        np.save(str(save_path / "valid_indices"), valid_indices)
        
    if analyze_errors:
        predictions = np.concatenate(predictions, 2).astype(dtype=np.float32) / 1000
        targets = np.concatenate(targets, 2).astype(dtype=np.float32) / 1000
        valid_indices = np.stack(valid_indices, -1)

        error_analysis(predictions, targets, valid_indices, settings)

    end_test_time = time.time()
    print("\nTotal time for evaluation is {} secs\n".format(end_test_time - start_test_time))


def load_settings() -> dict:
    """
    Returns:
        dict: The contents of settings.yaml as a dictionary
    """
    with open(Path(__file__).parent / "settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    return settings


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    help="Path to the folder containing the model to evaluate (overrides the the settings provided in settings.yaml)",
    type=Path,
)
parser.add_argument(
    "--test_data_path",
    help="Path to the folder containing the test dataframe (overrides the the settings provided in settings.yaml)",
    type=Path,
)
parser.add_argument(
    "--n_runs",
    help="Total number of validation runs (overrides the settings provided in settings.yaml)",
    type=int,
)
parser.add_argument(
    "--seed",
    help="Random seed (overrides the settings provided in settings.yaml)",
    type=int,
)
parser.add_argument(
    "--analyze-errors",
    help="Perform error analysis and upload the results to weights and biases (overrides the settings provided in settings.yaml)",
    type=bool,
)
parser.add_argument(
    "--save_results",
    help="Perform error analysis and upload the results to weights and biases (overrides the settings provided in settings.yaml)",
    type=bool,
    default=False,
)

args = parser.parse_args()
if __name__ == "__main__":
    settings = load_settings()

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(settings["script_settings"]["LOGGER_LEVEL"])

    model_path = args.model_path
    n_runs = args.n_runs
    path_to_test_df = args.test_data_path
    seed = args.seed
    analyze_errors = args.analyze_errors
    save_results = args.save_results

    if not args.model_path:
        model_path = settings["args"]["model_path"]
    else:
        settings["args"]["model_path"] = model_path

    if not args.n_runs:
        n_runs = settings["args"]["n_runs"]

    if not args.test_data_path:
        path_to_test_df = settings["args"]["test_data_path"]

    if not args.seed:
        seed = settings["args"]["seed"]

    if not args.analyze_errors:
        analyze_errors = settings["args"]["analyze_errors"]

    if analyze_errors:
        settings["wandb_custom"] = dict()
        settings["wandb_custom"]["run_name"] = input("Input a namefor this run: ")
        settings["wandb_custom"]["user_name"] = input("Who are you?: ")
        settings["wandb_custom"]["code_folder"] = model_path

    logger.info("RUNNING VALIDATION SCRIPT WITH SETTINGS:\n")
    logger.info(settings)

    evaluate(path_to_test_df, model_path, n_runs, settings, seed, analyze_errors, save_results)
