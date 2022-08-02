import torch
from torch import nn


def regressor_scores(preds, targets):
    if preds.shape != targets.shape:
        raise Exception("Different shapes between ground truths and predictions")

    _mae = nn.functional.l1_loss(targets, preds)
    _rmse = torch.sqrt(nn.functional.mse_loss(targets, preds))
    return _mae, _rmse


def filtered_scores(preds, targets, filter):
    # Select valid values.
    preds = preds[filter]
    targets = targets[filter]

    _mae, _rmse = regressor_scores(preds, targets)

    return _mae, _rmse


def baidu_loss_proxy(preds, targets, filter):

    _mae, _rmse = filtered_scores(preds, targets, filter)

    loss_proxy = 134 * (_mae + _rmse) / 2

    return loss_proxy
