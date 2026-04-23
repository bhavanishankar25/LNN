"""
Closed-form Continuous-depth (CfC) cell — the core of a Liquid Neural Network.

The underlying ODE for a Liquid Time-Constant neuron is:
    tau * dh/dt = -(h - A) * sigma(B)

where A (target state) and B (approach speed) are learned functions of the
input x and current hidden state h.

This ODE has an exact closed-form solution over a time interval dt:
    h(t + dt) = (1 - gate) * A + gate * h(t)
    gate = exp(-dt * sigma(B) / tau)

This is what makes CfC fast at inference: no numerical ODE solver needed,
just one evaluation of the closed-form expression per time step.
"""

import torch
import torch.nn as nn


class CfCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size

        # Maps [x, h] -> intermediate features
        self.backbone = nn.Linear(input_size + hidden_size, hidden_size)

        # A: attractor / target state
        self.fc_A = nn.Linear(hidden_size, hidden_size)

        # B: controls how fast hidden state approaches A
        self.fc_B = nn.Linear(hidden_size, hidden_size)

        # Learnable time constant per neuron (log-scale so tau is always positive)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,   # (batch, input_size)
        h: torch.Tensor,   # (batch, hidden_size)
        dt: torch.Tensor,  # (batch, 1) — time since last observation
    ) -> torch.Tensor:
        z = self.drop(torch.tanh(self.backbone(torch.cat([x, h], dim=-1))))

        A = self.fc_A(z)          # target state
        B = self.fc_B(z)          # approach speed

        tau = torch.exp(self.log_tau)                   # positive time constant
        gate = torch.exp(-dt * torch.sigmoid(B) / tau)  # decay toward A

        h_new = (1 - gate) * A + gate * h
        return h_new
