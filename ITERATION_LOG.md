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

### [2025-09-27] — Phase 2.1 dependencies + CI runnable

* Commit / Tag:
* Milestone: Phase 2 / M0
* Summary: Added requirements, install docs, and CI updates so torch and dependencies install cleanly.
* Changes:

  * requirements.txt: added runtime/test dependencies.
  * docs/INSTALL.md: CPU install/verify instructions.
  * docs/INSTALL_GPU.md: GPU install guidance for local 5090.
  * .github/workflows/ci.yml: ensure torch CPU wheel installed before other deps.
  * scripts/smoke_train.py: added CLI overrides for device/steps.
  * scripts/smoke_eval.py: added CLI override for device.
* Commands run:

  * Tests: `pytest -q`
  * Train: `python scripts/smoke_train.py -c configs/smoke.yaml --device cpu --steps 5`
  * Eval: `python scripts/smoke_eval.py -c configs/smoke.yaml --device cpu`
* Results:

  * Key metrics (SDR, DF, etc): N/A (dummy data)
  * Plots saved to: artifacts/smoke/smoke_eval_plot.png
* Issues / Bugs found: Previous Phase 2 tests failed due to missing torch dependency.
* Next steps: Continue Phase 2 M0 verification and expand data pipeline.
