from typing import Optional

import torch
from torch import nn


class FTJNF(nn.Module):
    def __init__(self, num_mics: int = 4, hidden_size: int = 32):
        super().__init__()
        input_dim = 2 * num_mics
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor, steering: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("expected input shape [B,T,F,2Q]")
        batch, frames, freq, channels = x.shape
        x_flat = x.reshape(batch * frames * freq, channels)
        out = self.proj(x_flat)
        out = out.reshape(batch, frames, freq, 2)
        return out
