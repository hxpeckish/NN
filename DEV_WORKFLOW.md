# Development Workflow (Rules for Codex + Contributors)

This project is an engineering-grade research reproduction. Follow these rules strictly.

---

## 1) Phase Discipline

* Phase 1: **spec files only** (already done)
* Phase 2: project skeleton + M0 dry-run (dummy data only)
* Phase 3+: real data + training + paper milestones

Never skip phases.

---

## 2) Small Modules Policy

* Prefer many small files over few large files.
* Any single file should be <= ~200 lines. If it grows, split it.
* Each module must have at least one unit test.

---

## 3) Test-First / Test-Always

Minimum required tests:

* STFT/iSTFT round-trip
* DMA pattern normalization at steering direction
* Complex mask application shape/multiplication correctness
* Dataset sample shapes and determinism under fixed seeds
* Metrics sanity checks (SDR, pattern aggregation, DF)

CI must run:

* `pytest -q`
* `python scripts/smoke_train.py --steps 5 --device cpu`

---

## 4) Config-First

* No hard-coded experiment params in Python.
* Everything must be in YAML under `configs/`.
* Scripts accept `-c path/to/config.yaml`.

---

## 5) Reproducible Data

* Any dataset generation must output a `manifest.json` containing:

  * DOAs, distances, room dims, RT60, seeds, source file IDs, noise IDs, array diameter.
* Running the same generation with same seeds must produce identical manifests.

---

## 6) Environment Compatibility (Cloud + Local 5090)

* Cloud/CI: CPU-only small-data smoke tests must pass.
* GPU must be optional:

  * never hard-code "cuda:0"
  * always use `torch.cuda.is_available()` to select device
* Do not pin CUDA versions in code.
* Provide `scripts/print_env.py` to print torch/cuda/cudnn versions.

Local server (5090):

* Installation should follow PyTorch official wheel matching the server driver.
* Provide a `docs/INSTALL_GPU.md` describing how to select the correct torch build.

---

## 7) Git Hygiene

* Each commit/PR must include:

  * what changed
  * how to run (commands)
  * results (numbers/plots paths when relevant)
* Update `ITERATION_LOG.md` for every milestone-affecting change.

---

## 8) Artifacts Convention

* Save outputs under `artifacts/<date_or_run_id>/...`
* Always produce:

  * config snapshot
  * metrics.json
  * plots
  * training log
  * checkpoint path

---
