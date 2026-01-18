# Architecture and Module Contracts

This document defines the code architecture and the interfaces between modules.
Primary goal: small modules, easy unit testing, minimal coupling.

---

## 1. Repository Layout (Target)

ndf_repro/
configs/
ndf/
audio/
arrays/
patterns/
data/
models/
losses/
metrics/
viz/
utils/
scripts/
tests/
artifacts/

---

## 2. Core Module Contracts

### 2.1 audio/stft.py

Responsibilities:

* STFT / iSTFT wrappers with fixed parameters from config.

Interfaces:

* `stft(wav: Tensor[B,T]) -> ComplexTensor[B,Frames,F]`
* `istft(spec: ComplexTensor[B,Frames,F]) -> Tensor[B,T]`

Unit tests:

* round-trip reconstruction error within tolerance.

---

### 2.2 arrays/geometry.py

Responsibilities:

* provide microphone positions (meters) for UCA+center.
  Interfaces:
* `uca_with_center(diameter_m: float) -> Tensor[Q,3]`
  Unit tests:
* center is (0,0,0)
* 3 outer mics are on circle of radius=diameter/2.

---

### 2.3 patterns/dma.py

Responsibilities:

* implement DMA-like directivity patterns with coefficients.
  Interfaces:
* `dma_pattern(theta_rad: Tensor[...,], order: int, coeffs: list[float], theta_s_rad: float) -> Tensor[...]`
  Unit tests:
* `pattern(theta_s)=1` (within tolerance)
* coefficients match REPRO_SPEC.md.

---

### 2.4 data/scene_simulator.py

Responsibilities:

* sample scene parameters (DOAs, distances, room dims, RT60, seeds)
* produce a manifest record for reproducibility.

Interfaces:

* `sample_scene(config, rng) -> dict`
* `write_manifest(records: list[dict], path)`

Unit tests:

* DOAs are within candidate set
* seeds stored and stable.

---

### 2.5 data/rir_habets.py

Responsibilities:

* generate RIRs (anechoic or reverberant) using a selected backend.
  Interfaces:
* `generate_rir(room, mic_positions, src_position, rt60, reflection_order) -> Tensor[Q, rir_len]`

Unit tests:

* reflection_order=0 produces a single direct-like peak (approx check).

---

### 2.6 data/dataset_builder.py

Responsibilities:

* load source wavs
* convolve with RIRs
* add self-noise
* loudness normalize
* produce training samples for model.

Interfaces:

* `build_sample(scene_record) -> dict` containing:

  * `y_mics_stft`: Tensor[T,F,2Q]
  * `z_target_stft`: Tensor[T,F,2]
  * metadata (doas, etc.)

Unit tests:

* shape checks
* deterministic output given fixed seeds.

---

### 2.7 models/ft_jnf.py

Responsibilities:

* FT-JNF network implementation producing complex mask.

Interfaces:

* `forward(x: Tensor[B,T,F,2Q], steering: Optional[Tensor[B,...]]) -> Tensor[B,T,F,2]` (mask)

Unit tests:

* output shape checks
* forward pass on dummy input.

---

### 2.8 losses/

* `l1_norm_batch.py`: implements batch-aggregated normalized L1
* `sdr_sa_t.py`: optional SDR-based loss

Unit tests:

* loss finite
* loss decreases on simple synthetic case.

---

### 2.9 metrics/

* `sdr.py`: aggregated SDR
* `power_pattern.py`: implements eq(10)-(14) estimation protocol
* `directivity_factor.py`: implements eq(17)

Unit tests:

* known synthetic cases produce expected results (sanity tests).

---

### 2.10 viz/

* `plot_pattern.py`: wideband polar + narrowband heatmap
* `plot_df.py`: DF vs frequency

Unit tests:

* functions run and save files.

---

## 3. Script Contracts

### scripts/train.py

* reads YAML config
* loads dataset (dummy or real)
* trains model
* saves checkpoint + logs

### scripts/eval.py

* loads checkpoint
* runs metrics
* outputs plots + json summary

### scripts/bandpass_analysis.py

* runs bandpass scenarios and outputs plots

---

## 4. Config Rules (YAML)

* All experiment parameters must be in YAML.
* No hard-coded paths; use a `data_root` in config.
* All random seeds are specified in config and written into manifest.

---
