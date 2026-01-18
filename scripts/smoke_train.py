import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ndf.data.dummy_dataset import DummyConfig, DummyDataset, collate_batch
from ndf.losses.l1_norm_batch import BatchNormalizedL1
from ndf.models.complex_mask import apply_complex_mask
from ndf.models.ft_jnf import FTJNF


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    torch.manual_seed(int(config.get("seed", 0)))

    device = torch.device(config.get("device", "cpu"))
    artifact_dir = Path(config.get("artifact_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dummy_cfg = DummyConfig(
        num_mics=int(config.get("num_mics", 4)),
        frames=int(config.get("dummy_frames", 10)),
        freqs=int(config.get("dummy_freqs", 257)),
    )
    dataset = DummyDataset(num_samples=int(config.get("dummy_samples", 20)), config=dummy_cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collate_batch,
    )

    model = FTJNF(num_mics=dummy_cfg.num_mics).to(device)
    criterion = BatchNormalizedL1()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))

    log_path = artifact_dir / "smoke_train_log.jsonl"
    steps = int(config.get("steps", 5))

    model.train()
    step = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        while step < steps:
            for batch in loader:
                x = batch["x"].to(device)
                y_ref = batch["y_ref"].to(device)
                z_target = batch["z_target"].to(device)

                mask = model(x)
                z_hat = apply_complex_mask(mask, y_ref)
                loss = criterion(z_hat, z_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                record = {
                    "step": step,
                    "loss": float(loss.detach().cpu().item()),
                    "x_shape": list(x.shape),
                    "mask_shape": list(mask.shape),
                }
                log_handle.write(json.dumps(record) + "\n")
                print(f"step={step} loss={record['loss']:.6f} x={record['x_shape']} mask={record['mask_shape']}")

                step += 1
                if step >= steps:
                    break

    ckpt_path = artifact_dir / "smoke_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
        },
        ckpt_path,
    )
    print(f"checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
