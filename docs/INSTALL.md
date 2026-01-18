# Installation (CPU)

## Requirements

* Python 3.10+

## Install

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Verify

```bash
pytest -q
python scripts/smoke_train.py -c configs/smoke.yaml --device cpu --steps 5
python scripts/smoke_eval.py -c configs/smoke.yaml --device cpu
```
