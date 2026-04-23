"""
LiquidNet — a sequence model built from stacked CfC cells.

Processes variable-length sequences with explicit time gaps (dt) between
observations. If dt is not provided, uniform spacing of 1.0 is assumed.
"""

import torch
import torch.nn as nn
from .cell import CfCCell


class LiquidNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(CfCCell(in_sz, hidden_size, dropout=dropout))

        self.readout = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,               # (batch, seq_len, input_size)
        dt: torch.Tensor | None = None, # (batch, seq_len, 1)
    ) -> torch.Tensor:                  # (batch, seq_len, output_size)
        batch_size, seq_len, _ = x.shape

        if dt is None:
            dt = torch.ones(batch_size, seq_len, 1, device=x.device)

        # Initialize hidden states to zero
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            dt_t = dt[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i], dt_t)
                inp = h[i]  # feed this layer's output as next layer's input
            outputs.append(self.readout(h[-1]))

        return torch.stack(outputs, dim=1)
