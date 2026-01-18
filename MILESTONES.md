# Milestones and Acceptance Criteria

This project reproduces key results of the NDF paper. Work proceeds in milestones. Each milestone must have:

* a runnable command
* produced artifacts (logs/plots)
* a short entry in `ITERATION_LOG.md`

---

## M0 — Project Skeleton + Dry-Run (No Real Data)

**Goal:** codebase runs end-to-end on dummy data on CPU.

**Acceptance:**

* `pytest` passes
* `python scripts/smoke_train.py --steps 5 --device cpu` runs and produces:

  * a checkpoint file
  * a training log
* `python scripts/smoke_eval.py` runs and produces:

  * at least one plot in `artifacts/`

---

## M1 — Anechoic, Static, 1st-Order Pattern (Minimum Repro Loop)

**Goal:** train NDF on anechoic simulated data for 1st-order cardioid at θs=0 and produce power pattern plots + SDR.

**Acceptance:**

* Train command (example):

  * `python scripts/train.py -c configs/anechoic_static_1st.yaml`
* Eval command:

  * `python scripts/eval.py -c configs/anechoic_static_1st.yaml --split test`
* Artifacts:

  * wideband polar power pattern
  * narrowband DOA×freq heatmap
  * SDR number for NDF and oracle param baseline
* Expected: NDF SDR > oracle param baseline (trend; exact values may vary)

---

## M1b — Anechoic, Static, Higher-Order (3rd + 6th)

**Goal:** reproduce the trend that higher-order patterns are learnable and L1 loss improves performance vs SDR-loss.

**Acceptance:**

* Train/eval for 3rd and 6th order patterns.
* Artifacts:

  * power pattern plots for 3rd and 6th
  * table of SDR for:

    * oracle param baseline
    * NDF (L1)
    * NDF (SDR-loss optional)
* Expected: NDF (L1) >= NDF (SDR-loss) for 3rd/6th (trend)

---

## M2 — Bandpass Aliasing Analysis

**Goal:** reproduce the key qualitative outcome of the bandpass experiments (Fig.6 behavior).

**Acceptance:**

* Script:

  * `python scripts/bandpass_analysis.py -c configs/bandpass.yaml`
* Must implement 4 cases:

  1. 1 kHz band only -> pattern OK
  2. 7 kHz band only -> aliasing / distorted pattern
  3. 7 kHz + full spectrum below aliasing freq -> pattern OK at 7 kHz
  4. 1 kHz + 7 kHz only -> aliasing returns at 7 kHz
* Artifacts:

  * plots for these cases (heatmaps + narrowband slices)

---

## M3 — Reverberant: A-Model vs R-Model + DF

**Goal:** reproduce the trend that R-Model improves SDR and DF in reverberation.

**Acceptance:**

* Train A-Model (anechoic only) and evaluate on reverberant test.
* Train R-Model (reverb + small anechoic mix) and evaluate.
* Artifacts:

  * SDR table vs RT60 variants (at least 0.2s and 0.6s)
  * DF vs frequency plots
  * wideband power patterns showing improved suppression for higher-order under stronger reverb (trend)

---

## M4 — Steerable + User-Defined Patterns (Optional Extension)

**Goal:** implement steering conditioning and demonstrate directional control and user-defined patterns.

**Acceptance:**

* Steerable model:

  * evaluate at θs ∈ {0,30,60,90,120} (trained directions)
  * produce SDR table and power patterns for at least two directions
* User-defined patterns:

  * define two target patterns
  * train and show wideband/narrowband patterns approximated

---
