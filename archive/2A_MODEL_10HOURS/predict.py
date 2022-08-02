from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataset import Scaler
from model import BaselineGruModel
from pl_wrapper import PLWrapper
from pytorch_lightning.utilities import seed
seed.seed_everything(2022, True)


def forecast_kfold(input_data,settings,run_name):
    model = BaselineGruModel(settings, cuda=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler_base_path = Path(__file__).parent / f"checkpoints_{run_name}_10HOURS" / "scalers"
    model_path = Path(__file__).parent / f"checkpoints_{run_name}_10HOURS" / f"model_{run_name}.ckpt" 

    scaler = Scaler.load(scaler_base_path, f"{run_name}_ins", device)
    tgt_scaler = Scaler.load(scaler_base_path, f"{run_name}_tgt", device)

    model_wrapper = PLWrapper.load_from_checkpoint(model_path, model=model).to(device)
    model.device = device

    inputs = list()

    with torch.no_grad():
        # Create batches

        for TurbID, group in input_data[settings["inputs"]].groupby(input_data.TurbID):
            input_sample = (
                torch.from_numpy(group.iloc[-settings["input_length"] :].to_numpy())
                .unsqueeze(0)
                .to(dtype=torch.float32, device=device)
            )

            inputs.append(input_sample)

        X = torch.concat(inputs)

        X_scaled = scaler.transform(X)

        out = model(X_scaled)
        out_descaled = tgt_scaler.inverse_transform(out)

    predictions = out_descaled.clip(0, 1500)

    assert predictions.shape == (134, 60, 1)

    return predictions.cpu().numpy()

def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions as a tensor \in R^{134 * 288 * 1}
    """
    input_data = None
    if "custom_evaluation_dataframe" in settings.keys():
        input_data = settings["custom_evaluation_dataframe"]
    else:
        input_data = pd.read_csv(settings["path_to_test_x"])

    pred_1 =  forecast_kfold(input_data,settings,"fold-1")
    pred_2 =  forecast_kfold(input_data,settings,"fold-2")
    pred_3 =  forecast_kfold(input_data,settings,"fold-3")
    pred_4 =  forecast_kfold(input_data,settings,"fold-4")
    pred_5 =  forecast_kfold(input_data,settings,"fold-5")

    pred_final_hora = np.max([pred_1, pred_2, pred_3, pred_4, pred_5], axis=0)
    print(pred_1)
    asdsadasd
    return pred_final_hora

    
