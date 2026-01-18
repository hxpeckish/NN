# Installation (GPU)

## Requirements

* Python 3.10+
* NVIDIA GPU + drivers installed (verify with `nvidia-smi`)

## Install

1. Check your CUDA driver version with `nvidia-smi`.
2. Select the matching PyTorch CUDA wheel from the official installer: https://pytorch.org/get-started/locally/.
3. Install the selected wheel, then install project requirements:

```bash
python -m pip install --upgrade pip
# Example only; follow the official selector for your CUDA version
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Verify

```bash
python scripts/print_env.py
pytest -q
python scripts/smoke_train.py -c configs/smoke.yaml --device cuda --steps 5
python scripts/smoke_eval.py -c configs/smoke.yaml --device cuda
```
