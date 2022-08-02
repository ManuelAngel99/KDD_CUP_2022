from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from dataset import Scaler
from model import BaselineGruModel
from pytorch_lightning.utilities import seed
seed.seed_everything(2022, True)

def forecast_kfold(input_data,settings,run_name):
    from pl_wrapper import PLWrapper
    from model import BaselineGruModel
    model = BaselineGruModel(settings, cuda=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler_base_path = Path(__file__).parent / f"checkpoints_{run_name}" / "scalers"
    model_path = Path(__file__).parent / f"checkpoints_{run_name}" / f"model_{run_name}.ckpt" 

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

    assert predictions.shape == (134, 288, 1)

    return predictions.cpu().numpy()



def forecast_kfold_hora(input_data,settings,run_name):
    from pl_wrapper_10HOURS import PLWrapperHora
    from model_10HOURS import BaselineGruModel

    model = BaselineGruModel(settings, cuda=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler_base_path = Path(__file__).parent / f"checkpoints_{run_name}_10HOURS" / "scalers"
    model_path = Path(__file__).parent / f"checkpoints_{run_name}_10HOURS" / f"model_{run_name}.ckpt" 

    scaler = Scaler.load(scaler_base_path, f"{run_name}_ins", device)
    tgt_scaler = Scaler.load(scaler_base_path, f"{run_name}_tgt", device)

    model_wrapper = PLWrapperHora.load_from_checkpoint(model_path, model=model).to(device)
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

# modelo 1 hora
def forecast_kfold_una_hora(input_data,settings,run_name):
    from pl_wrapper_1HOURS import PLWrapperUnaHora
    from model_1HOURS import BaselineGruModel
    model = BaselineGruModel(settings, cuda=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler_base_path = Path(__file__).parent / f"checkpoints_{run_name}_1HOURS" / "scalers"
    model_path = Path(__file__).parent / f"checkpoints_{run_name}_1HOURS" / f"model_{run_name}.ckpt" 

    scaler = Scaler.load(scaler_base_path, f"{run_name}_ins", device)
    tgt_scaler = Scaler.load(scaler_base_path, f"{run_name}_tgt", device)

    model_wrapper = PLWrapperUnaHora.load_from_checkpoint(model_path, model=model).to(device)
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
    assert predictions.shape == (134, 6, 1)

    return predictions.cpu().numpy()

# modelo 3 horas
def forecast_kfold_tres_hora(input_data,settings,run_name):
    from pl_wrapper_3HOURS import PLWrapperTresHora
    from model_3HOURS import BaselineGruModel
    model = BaselineGruModel(settings, cuda=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler_base_path = Path(__file__).parent / f"checkpoints_{run_name}_3HOURS" / "scalers"
    model_path = Path(__file__).parent / f"checkpoints_{run_name}_3HOURS" / f"model_{run_name}.ckpt" 

    scaler = Scaler.load(scaler_base_path, f"{run_name}_ins", device)
    tgt_scaler = Scaler.load(scaler_base_path, f"{run_name}_tgt", device)

    model_wrapper = PLWrapperTresHora.load_from_checkpoint(model_path, model=model).to(device)
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
    assert predictions.shape == (134, 18, 1)

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

    # modelo 48 horas
    pred_1 =  forecast_kfold(input_data,settings,"fold-1")
    pred_2 =  forecast_kfold(input_data,settings,"fold-2")
    pred_3 =  forecast_kfold(input_data,settings,"fold-3")
    pred_4 =  forecast_kfold(input_data,settings,"fold-4")
    pred_5 =  forecast_kfold(input_data,settings,"fold-5")
    pred_final = (pred_1+pred_2+pred_3+pred_4+pred_5)/5
    pred_final = np.max([pred_1, pred_2, pred_3, pred_4, pred_5], axis=0)

    # modelo 10 horas
    pred_final_hora_1 = forecast_kfold_hora(input_data,settings,"fold-1")
    pred_final_hora_2 = forecast_kfold_hora(input_data,settings,"fold-2")
    pred_final_hora_3 = forecast_kfold_hora(input_data,settings,"fold-3")
    pred_final_hora_4 = forecast_kfold_hora(input_data,settings,"fold-4")
    pred_final_hora_5 = forecast_kfold_hora(input_data,settings,"fold-5")
    pred_final_hora = (pred_final_hora_1+pred_final_hora_2+pred_final_hora_3+pred_final_hora_4+pred_final_hora_5)/5
    pred_final_hora = np.max([pred_final_hora_1, pred_final_hora_2, pred_final_hora_3, pred_final_hora_4, pred_final_hora_5], axis=0)
    
    # modelo  3 horas
    pred_1_tres_hora = forecast_kfold_tres_hora(input_data,settings,"fold-1")
    pred_2_tres_hora = forecast_kfold_tres_hora(input_data,settings,"fold-2")
    pred_3_tres_hora = forecast_kfold_tres_hora(input_data,settings,"fold-3")
    pred_4_tres_hora = forecast_kfold_tres_hora(input_data,settings,"fold-4")
    pred_5_tres_hora = forecast_kfold_tres_hora(input_data,settings,"fold-5")
    pred_final_tres_hora = (pred_1_tres_hora+pred_2_tres_hora+pred_3_tres_hora+pred_4_tres_hora+pred_5_tres_hora)/5
    pred_final_tres_hora = np.max([pred_1_tres_hora,pred_2_tres_hora,pred_3_tres_hora,pred_4_tres_hora,pred_5_tres_hora ], axis=0)
    # modelo 1 hora
    pred_1_una_hora = forecast_kfold_una_hora(input_data,settings,"fold-1")
    pred_2_una_hora = forecast_kfold_una_hora(input_data,settings,"fold-2")
    pred_3_una_hora = forecast_kfold_una_hora(input_data,settings,"fold-3")
    pred_4_una_hora = forecast_kfold_una_hora(input_data,settings,"fold-4")
    pred_5_una_hora = forecast_kfold_una_hora(input_data,settings,"fold-5")
    pred_final_tres_hora = (pred_1_una_hora+pred_2_una_hora+pred_3_una_hora+pred_4_una_hora+pred_5_una_hora)/5
    pred_final_tres_hora = np.max([pred_1_una_hora,pred_2_una_hora,pred_3_una_hora,pred_4_una_hora,pred_5_una_hora ], axis=0)
    

  #    
    # KNN
    df = pd.read_csv(os.path.join(
        settings["checkpoints"], "wtbdata_245days.csv"))

    # ELIMINAR CUANDO SUBAMOS A TEST.
    #df = df.drop(df[df.Day > 165].index)

    input_data = pd.concat([df, input_data])


    input_data = input_data.fillna(method="ffill")

    LF = 288
    LB = 6

    k = 2000

    assert LB < LF

    predictions = torch.zeros(134, 288, 1)

    for TurbID, group in input_data.groupby("TurbID").Patv:
        data = torch.tensor(group.to_numpy())

        # LB size(ej:6) historic values to compare with
        possible_neighbours = data[:-(LF)].unfold(dimension=0, size=LB, step=1)
        # LF size (288) values to predict
        possible_targets = data[LB:].unfold(dimension=0, size=LF, step=1)

        knn_input = data[-LB:]

        # Find the indices of the kNN
        values, indices = torch.topk(
            torch.sum((possible_neighbours - knn_input) ** 2, axis=1), k=k, largest=False
        )

        # Select the kNN and take their weighted avg
        tmp = 1 / (1 + values)
        weights = tmp / sum(tmp)

        # no habria que sumarle uno al index de targets?
        nearest_neighbours_targets = weights @ possible_targets[indices]

        # Save the result as a matrix
        predictions[TurbID - 1, :,
                    :] = nearest_neighbours_targets.unsqueeze(-1)

    predictions_knn = predictions.numpy()
    assert predictions_knn.shape == (134, 288, 1)

    
    pred_final = np.max([pred_final, predictions_knn], axis=0)
    pred_final[:,:60,:] = pred_final_hora
    pred_final[:,:18,:] = pred_1_tres_hora
    pred_final[:,:6,:] = pred_1_una_hora

    print(pred_final)
    asdasd
    return pred_final
    
