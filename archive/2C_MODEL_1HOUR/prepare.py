"""
    This script is required by the competition organizers. Its goal is to setup the environment
    for the model evaluation system with the prep_env function. In addition to that,
    the prep_env function will be used to store the model settings
    (as done in the baseline model released by Baidu).
"""


def prep_env():
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        # Required fields.
        "checkpoints": "checkpoints",
        "framework": "pytorch",
        "pred_file": "predict.py",
        "start_col": 3,
        # Additional settings
        "input_length": 144,
        "output_length": 60,
        "inputs": ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"],
        "targets": ["Patv"],
        "lstm_layer": 2,
        "dropout": 0.3,
        "train_epochs": 5,
        "train_batch_size": 256,
        "val_batch_size": 512,
        "lr": 1e-5,
        "is_debug": True,
        "weight_decay": 1e-4,
    }

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
