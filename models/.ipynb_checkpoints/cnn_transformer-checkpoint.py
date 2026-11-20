
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Adds positional information to the sequence."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, sequence_length, d_model)
        return x + self.pe[:, :x.size(1), :]


class CNNTransformer(nn.Module):
    """
    CNN-Transformer for workload forecasting.

    Input:  (batch, sequence_length, 1)
    Output: (batch, prediction_horizon)
    """

    def __init__(
        self,
        input_features=1,
        cnn_channels=16,
        kernel_size=3,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=128,
        dropout=0.1,
        prediction_horizon=1,
    ):
        super().__init__()

        # Extract local workload patterns
        self.cnn = nn.Conv1d(
            in_channels=input_features,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()

        # Project CNN features into Transformer embedding dimension
        self.projection = nn.Linear(cnn_channels, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # Predict the next workload value from the final timestep
        self.output_layer = nn.Linear(d_model, prediction_horizon)

    def forward(self, x):
        # (batch, sequence, features) -> (batch, features, sequence)
        x = x.transpose(1, 2)

        # CNN extracts local patterns
        x = self.relu(self.cnn(x))

        # (batch, channels, sequence) -> (batch, sequence, channels)
        x = x.transpose(1, 2)

        x = self.projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.norm(x)

        # Use the final timestep to forecast future workload
        x = x[:, -1, :]

        return self.output_layer(x)
