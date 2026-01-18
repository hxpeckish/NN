from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class DummyConfig:
    num_mics: int = 4
    frames: int = 10
    freqs: int = 257


class DummyDataset(Dataset):
    def __init__(self, num_samples: int, config: DummyConfig):
        self.num_samples = num_samples
        self.config = config

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shape_x = (self.config.frames, self.config.freqs, 2 * self.config.num_mics)
        x = torch.randn(*shape_x)
        y_ref = torch.randn(self.config.frames, self.config.freqs, 2)
        z_target = torch.randn(self.config.frames, self.config.freqs, 2)
        return {
            "x": x,
            "y_ref": y_ref,
            "z_target": z_target,
        }


def collate_batch(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    y_ref = torch.stack([item["y_ref"] for item in batch], dim=0)
    z_target = torch.stack([item["z_target"] for item in batch], dim=0)
    return {
        "x": x,
        "y_ref": y_ref,
        "z_target": z_target,
    }
