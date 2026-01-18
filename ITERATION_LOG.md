# Iteration Log

This file records the chronological engineering/research iterations.
Every milestone-affecting change must add an entry.

---

## Template (copy for each entry)

### [YYYY-MM-DD] — <short title>

* Commit / Tag:
* Milestone:
* Summary:
* Changes:

  * (module/file): what changed
* Commands run:

  * Train:
  * Eval:
  * Tests:
* Results:

  * Key metrics (SDR, DF, etc):
  * Plots saved to:
* Issues / Bugs found:
* Next steps:

---

## Entries

### [INIT] — Spec and workflow added

* Commit / Tag:
* Milestone: Phase 1
* Summary: Added reproduction spec + milestones + architecture + workflow + log template.
* Next steps: Phase 2 skeleton + M0 dry-run.

---

### [2025-09-27] — Phase 2 skeleton + M0 smoke

* Commit / Tag:
* Milestone: Phase 2 / M0
* Summary: Added project skeleton, dummy data pipeline, smoke train/eval scripts, and CI.
* Changes:

  * ndf/audio/stft.py: STFT/iSTFT utilities with configurable params.
  * ndf/models/ft_jnf.py: minimal FT-JNF placeholder model.
  * ndf/models/complex_mask.py: complex mask application.
  * ndf/losses/l1_norm_batch.py: batch-normalized L1 loss.
  * ndf/data/dummy_dataset.py: dummy dataset for smoke tests.
  * scripts/smoke_train.py: 5-step CPU smoke train.
  * scripts/smoke_eval.py: smoke eval and plot.
  * scripts/print_env.py: environment reporting.
  * configs/smoke.yaml: smoke configuration.
  * tests/: minimal unit tests.
  * .github/workflows/ci.yml: CI for pytest + smoke train.
* Commands run:

  * Train: `python scripts/smoke_train.py -c configs/smoke.yaml`
  * Eval: `python scripts/smoke_eval.py -c configs/smoke.yaml`
  * Tests: `pytest -q`
* Results:

  * Key metrics (SDR, DF, etc): N/A (dummy data)
  * Plots saved to: artifacts/smoke/smoke_eval_plot.png
* Issues / Bugs found: None.
* Next steps: Phase 2 M0 verification on CI, prepare real-data pipeline.
