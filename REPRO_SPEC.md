# NDF Reproduction Specification (Paper-to-Code Contract)

This document is the **source of truth** for reproducing the paper:
**"Neural Directional Filtering Using a Compact Microphone Array"** (Huang et al., 2025).

Any implementation must follow this spec unless explicitly amended via a logged update in `ITERATION_LOG.md`.

---

## 1. Goal

Reproduce the key effects reported in the paper:

* Neural Directional Filtering (NDF) produces an output approximating a **Virtual Directional Microphone (VDM)** with a **predefined directivity pattern**.
* NDF can learn **1st / 3rd / 6th-order** DMA-like patterns using a **compact 4-microphone array**.
* NDF can achieve **frequency-invariant** patterns and mitigate **spatial aliasing** for broadband signals.
* In reverberation, an R-Model (trained with reverberant data) improves SDR/DF vs an A-Model (trained only on anechoic).
* (Optional) Steerable patterns via steering direction conditioning (one-hot).

---

## 2. Array Geometry

### 2.1 Microphones

* Number of microphones: **Q = 4**
* Geometry: **3 microphones on a Uniform Circular Array (UCA)** + **1 microphone at the center**
* Reference microphone: **center microphone** (unless a specific experiment explicitly uses another reference)

### 2.2 Default diameters

* Default UCA diameter: **3 cm**
* Additional experiments: **6 cm** and **9 cm** (array aperture study)

All geometry must be implemented in meters.

---

## 3. Signal Representation (STFT)

* Sampling rate: **16 kHz**
* STFT window length: **32 ms**
* Window: **square-root Hann**
* Overlap: **50%**
* Complex STFT values are used.
* Input feature tensor: stack real and imag parts of each mic signal

  * Shape convention: **[B, T, F, 2Q]**
  * Channel order must be documented and consistent.

Output masking is complex and applied to the reference mic STFT:

* **Z_hat[f,t] = M[f,t] * Y_ref[f,t]**
* where **M[f,t]** is a complex mask.

---

## 4. Target Directivity / VDM Target Signal

### 4.1 DMA pattern form

Target pattern follows the paper’s DMA-style pattern definition (conceptually):

* Use coefficients (a_j) to define the pattern order J.

Use the exact coefficients below (from Table I of the paper):

* **1st-order**: {a0=1/2, a1=1/2}
* **3rd-order**: {a0=0, a1=1/6, a2=1/2, a3=1/3}
* **6th-order**: {a0=1/49, a1=8/49, a2=8/49, a3=-48/49, a4=-48/49, a5=64/49, a6=64/49}

Steering direction:

* Default static experiments: **θs = 0°**
* Steering is restricted to the x-y plane (elevation φ = 0 for reported visualizations).

### 4.2 Target VDM signal synthesis

* Anechoic: use only direct-path component (paper eq. (5))
* Reverberant: VDM incorporates reflections weighted by the directivity pattern (paper eq. (3)/(4))

Implementation note:

* For anechoic VDM target, it is sufficient to synthesize the VDM signal by applying the pattern gain Λ(θ_n) to each source at the reference position (with direct-path transfer), and summing sources.
* For reverberant, the RIR-based synthesis should apply direction-dependent weighting to reflection paths as described in the paper; if exact path-wise decomposition is not feasible, document the approximation used and ensure evaluation is consistent.

### 4.3 Numerical stability

* Maximum attenuation cap: **40 dB** (linear floor: **0.01**) for target patterns.
* Epsilon used in losses/metrics: **1e-7**

---

## 5. Model Architecture (FT-JNF)

Use the paper’s FT-JNF (based on [34]):

* F-BiLSTM: **bidirectional**, hidden size **256**

  * Operates along the frequency dimension (instantaneous spectro-spatial modeling)
* T-UniLSTM: **unidirectional**, hidden size **128**

  * Operates causally along time, treating frequency as batch
* Output head:

  * Linear + **tanh**, then Linear
  * Outputs a **complex single-channel mask** with 2 channels (real/imag)
* Parameter count target: approximately **0.873M** trainable parameters (sanity check; small deviations allowed if clearly justified)

### 5.1 Optional steering mechanism (steerable patterns)

* Steering angle θs encoded as **one-hot** vector
* Angular resolution: **ϑ = 5°** => M = 72 discrete steering directions
* One-hot -> Linear embedding -> merged into network as in the paper (document exact merge: add/concat)

Note: one-hot conditioning restricts inference to trained steering directions; this is acceptable for reproduction.

---

## 6. Loss Functions

Default loss: **batch-aggregated normalized L1** (paper eq. (8))

Also implement for ablation:

* SA-ε-tSDR style loss (paper eq. (7)) as optional comparison.

---

## 7. Training Strategy & Sampling

### 7.1 Training mixture sizes

* Number of concurrent sources per scene during training: **N ∈ {1,2,3}**
* Models trained with up to 3 speakers should generalize to more (not required for core reproduction).

### 7.2 Mini-batch sampling constraint (important)

To stabilize training:

* Each mini-batch must contain **at least one sample** with a source at or near the target direction:

  * within **±20°** of steering direction.

---

## 8. Data & Datasets

### 8.1 Source audio

* Train: **LibriSpeech train-clean-360**
* Validation: **LibriSpeech dev-clean**
* Speech test: **EARS** dataset (apply loudness-based selection and extract a 4-second segment with above-average loudness)
* Non-speech test: **WHAM!** noise

All signals:

* Trim/pad to **4 seconds** (pad with zeros).
* For multi-source non-speech scenes, trim to the shortest source length (then pad if needed).

### 8.2 Loudness normalization

Normalize convolved signals to have loudness within:

* **[-33, -25] dBFS**

Implementation must document the loudness algorithm used. If ITU-R BS.1770-5 is used, document it; if an approximation is used, keep it consistent across all data.

### 8.3 Sensor self-noise

Add spatially uncorrelated white Gaussian noise to microphone signals.
Default SNR:

* **30 dB** relative to the mixture of all sources.

---

## 9. Anechoic Data Generation

* Fixed source-array distance: **d = 1.5 m**
* Candidate DOAs:

  * Train: P_train=72, θ ∈ {0°, 5°, ..., 355°}
  * Val: P_val=72, θ ∈ {2.5°, 7.5°, ..., 357.5°}
  * Test: P_test=144, θ ∈ {1.25°, 3.75°, ..., 358.75°}
* For each scene, randomly sample N source DOAs from candidate set.
* RIR:

  * Use Habets RIR generator with **reflection order = 0** (direct-path only).

Anechoic test sets:

* Each sample contains **two concurrent speakers**.
* Static pattern test sample count: **3240**
* Steerable test: create 5 steering directions θs ∈ {0°, 30°, 60°, 90°, 120°}, total **3240×5** samples.

---

## 10. Reverberant Data Generation

Room parameter ranges (Table II):

* Length: 6–10 m
* Width: 4–8 m
* Height: 3–5 m
* RT60: 0.2–0.5 s
* Source-array distance: 0.5–2.5 m

Placement:

* Place array position ensuring **>= 1.2 m** from each wall.

Train/val/test DOA candidate sets are same as anechoic.

Static pattern training sample counts (for θs=0):

* Train: **52880** (50000 reverberant + 2800 anechoic)
* Val: **6360** (6000 reverberant + 360 anechoic)
* Test: **3240** reverberant samples
* Test samples contain **two concurrent speakers**.

---

## 11. Training Hyperparameters

* Batch size: **10**
* Static patterns (anechoic): max **250 epochs**, LR **1e-3**
* Steerable patterns (anechoic): max **100 epochs**, LR **1e-3**, reduce by factor **0.75 after 50 epochs**
* Static patterns (reverberant): max **150 epochs**
* Model selection: checkpoint with **lowest validation loss**

---

## 12. Evaluation Metrics

Implement:

* SDR (aggregated) as in paper eq. (19)
* Power pattern estimation (paper eq. (10)-(14)):

  * Apply estimated mask to **direct-path components** to estimate power ratios
  * Aggregate by DOA over the full test set
  * Provide narrowband (DOA×freq heatmap) and wideband (polar) patterns
* Directivity Factor DF (paper eq. (17)):

  * Compute using reverberant components

Visualization deliverables:

* Wideband power pattern polar plot
* Narrowband power pattern DOA×frequency heatmap
* DF vs frequency plot

---

## 13. Baselines

Implement at least:

1. Oracle parametric masking baseline:

   * Compute real-valued mask G[f,t] by evaluating target pattern using oracle DOA
   * Apply: Z_b[f,t] = G[f,t] * Y_ref[f,t]

2. LS beamformer baseline:

   * Include if feasible; if not, clearly state limitations and include oracle param baseline as primary baseline.

---

## 14. Reproducibility Requirements

* All experiments must be seed-controlled:

  * separate seeds for scene sampling, source sampling, noise sampling, RIR sampling.
* Every generated dataset must include a `manifest.json` capturing:

  * DOAs, distances, room dims, RT60, seeds, source file ids, noise ids.
* No hard-coded absolute paths.

---
