import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml

from ndf.data.dummy_dataset import DummyConfig, DummyDataset, collate_batch
from ndf.models.complex_mask import apply_complex_mask
from ndf.models.ft_jnf import FTJNF

matplotlib.use("Agg")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = torch.device(config.get("device", "cpu"))
    artifact_dir = Path(config.get("artifact_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dummy_cfg = DummyConfig(
        num_mics=int(config.get("num_mics", 4)),
        frames=int(config.get("dummy_frames", 10)),
        freqs=int(config.get("dummy_freqs", 257)),
    )
    dataset = DummyDataset(num_samples=2, config=dummy_cfg)
    batch = collate_batch([dataset[0], dataset[1]])
    x = batch["x"].to(device)
    y_ref = batch["y_ref"].to(device)

    model = FTJNF(num_mics=dummy_cfg.num_mics).to(device)
    ckpt_path = Path(config.get("checkpoint_path", artifact_dir / "smoke_checkpoint.pt"))
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    model.eval()
    with torch.no_grad():
        mask = model(x)
        z_hat = apply_complex_mask(mask, y_ref)

    magnitude = torch.sqrt(z_hat[..., 0] ** 2 + z_hat[..., 1] ** 2)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(magnitude[0].cpu().numpy(), aspect="auto", origin="lower")
    ax.set_title("Dummy Output Magnitude")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Frame")
    fig.tight_layout()

    plot_path = artifact_dir / "smoke_eval_plot.png"
    fig.savefig(plot_path)
    print(f"plot saved to {plot_path}")


if __name__ == "__main__":
    main()
