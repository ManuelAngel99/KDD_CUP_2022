import torch
from torch import nn as nn


class BaselineGruModel(nn.Module):
    """
    Desc:
        A simple GRU model
    """

    def __init__(self, settings: dict, cuda=False) -> None:
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """

        super().__init__()
        self.output_len = settings["output_length"]
        self.hidC = len(settings["inputs"])
        self.hidR = 48
        self.out_dim = len(settings["targets"])
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(
            input_size=self.hidC,
            hidden_size=self.hidR,
            num_layers=settings["lstm_layer"],
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidR, self.out_dim)

        self.device = "cuda" if (torch.cuda.is_available() and cuda == True) else "cpu"

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc: Input tensor
        Returns:
            Model forecasts
        """

        x = torch.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]], device=self.device)
        x_enc = torch.concat((x_enc, x), 1)

        # GRU + linear projection
        dec, _ = self.lstm(x_enc)
        forecast = self.projection(self.dropout(dec))

        # Select the outputs from the sequence.
        forecast = forecast[:, -self.output_len :, -self.out_dim :]

        return forecast  # [B, L, D]
