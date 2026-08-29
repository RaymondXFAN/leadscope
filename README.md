# LeadScope — Dyadic Dance Leader-Follower Coupling & Lag Analysis

A methodological toolkit for analysing **leader-follower temporal coupling** and **frequency-band synchrony** in dyadic dance motion capture data, built on the [Bigand et al. (2024) Dyadic-dance-coordination](https://github.com/TheoPulse/Dyadic-dance-coordination) dataset.

This repository accompanies the manuscript *"Coupling Structure and Lag Estimation in Dyadic Dance Motion: A Time–Frequency Approach on the Bigand Dataset"*. It provides the full analysis pipeline — from raw velocity signals to statistical inference of leader-follower relationships, music-symmetry effects, and frequency-band-specific coupling.

---

## 1. Software & Hardware Environment

### Operating System
- Linux (tested on Ubuntu 20.04 / AutoDL container) or macOS
- Windows untested but should work with WSL2

### Python Runtime
- **Python 3.12+** (tested on 3.13.12)
- No GPU required (all analyses are CPU-based, multi-core parallel via `joblib`)

### Python Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Array computation |
| `scipy` | ≥1.11 | `scipy.signal.coherence`, `scipy.stats` (Mann-Whitney, Friedman, chi-square) |
| `joblib` | ≥1.3 | Parallel estimation across windows/sub-segments |
| `torch` | ≥2.0 | Only for loading the Bigand `.pkl` (data stored as tensors; auto-converted to numpy) |

Install:
```bash
pip install numpy scipy joblib torch
```

> **Note on `torch`**: The Bigand dataset stores arrays as PyTorch tensors. `lagcc` and all scripts auto-convert to numpy via `to_np()`. If you convert the data to numpy `.pkl` beforehand, `torch` is unnecessary.

---

## 2. Data Acquisition

### 2.1 Source Dataset

The motion capture data comes from:

> Bigand, Félix, et al. "The geometry of interpersonal synchrony in human dance." *Current Biology* 34.13 (2024): 3011-3019.e4.

The analysis code repository is public:
```bash
git clone https://github.com/TheoPulse/Dyadic-dance-coordination.git
cd Dyadic-dance-coordination
```

### 2.2 Data Files Used

This study uses the **5 Hz downsampled** velocity signals (the only version publicly available on GitHub):

| File | Size | Condition | Used in |
|---|---|---|---|
| `dyadic_dance/data/both_cond_only/raw_velocities_ds.pkl` | 52.3 MB | Vision + No-vision combined | All main analyses |
| `dyadic_dance/data/both_cond_only/features_ds.pkl` | 66.6 MB | Pre-computed synchrony features | Not used (baseline reference) |
| `dyadic_dance/data/vis_only/raw_velocities_ds.pkl` | 52.5 MB | Vision-only condition | Cross-validation (§4.4 of manuscript) |
| `dyadic_dance/data/vis_only/features_ds.pkl` | 66.8 MB | Vision-only features | Not used |

### 2.3 Data Format (confirmed via source-code reverse engineering)

Each `.pkl` is a **2-tuple `(X, y)`** in tsai convention:

- **`X`**: Tensor, shape `(N=558, C=132, T=186)` — 558 windows, 132 channels, 186 frames (37.2 s @ 5 Hz)
  - **Channel split**: `n_sensors = X.shape[1] // 2 = 66` → channels `[0:66]` = dancer A, `[66:132]` = dancer B (paired dyadic signal by construction)
- **`y`**: Tensor, shape `(558, 6)` — labels with 6 columns:

| Column | Name | Values | Meaning |
|---|---|---|---|
| 0 | `label` | {0, 1} | Coupling condition (0 = no-vision, 1 = vision) |
| 1 | `dyad_number` | 1–35 | Dyad identifier (35 unique dyads) |
| 2 | `trial` | 1–32 | Trial number |
| 3 | `song_left` | 0–7 | Music stimulus for dancer A (8 songs; Bigand uses different music to break symmetry) |
| 4 | `song_right` | 0–7 | Music stimulus for dancer B |
| 5 | `id` | 0–2207 | Unique window id |

### 2.4 The 25 Hz Data Issue (important)

The repository also contains `raw_velocities_ds25hz.pkl` (134 B) — these are **Git LFS pointer files**, not real data. The LFS objects were **never uploaded** (547 MB × 2 exceeds GitHub's free LFS quota). The `download_lfs_25hz.py` script attempts to fetch them via the LFS Batch API but the server returns *"Object does not exist on the server."*

**Conclusion**: Only the 5 Hz version is usable. This limits temporal-lag resolution to 200 ms/frame (see manuscript §5.1 Limitations). The original mocap data exists at [IIT Dataverse doi:10.48557/UR2GBG](https://doi.org/10.48557/UR2GBG) but requires manual downsampling — left as future work.

---

## 3. Software Architecture

```
leadscope/
├── src/
│   └── lagcc.py                      # Core: lagged cross-correlation library (v1.1)
├── scripts/
│   ├── probe_bigand_data.py          # Data structure inspection (diagnostic)
│   ├── inspect_bigand_repo.py        # Repository health check (diagnostic)
│   ├── download_bigand.py            # Dataset download helper
│   ├── download_lfs_25hz.py         # LFS Batch API downloader (25Hz, currently non-functional)
│   ├── pilot_lag_bigand.py           # Experiment 1: pilot lag estimation (v2.1, 4 representations)
│   ├── sw_subwindow_lag.py           # Experiment 2: sliding-window time-varying lag
│   ├── song_influence_lag.py         # Experiment 3: music-symmetry effect on coupling
│   ├── coherence_analysis.py         # Experiment 4: frequency-band coherence (v2: nperseg=32, dyad output)
│   ├── dyad_aggregate_stats.py       # Dyad-level (N=35) confirmatory statistics (Wilcoxon/FDR/bootstrap)
│   └── make_figures.py               # Paper figures: Morandi palette + TNR ≥12pt, no captions
├── results/                          # Output CSVs (auto-created)
├── figures/                          # Paper figures (PNG, 200 dpi)
├── README.md                         # This file (deployment manual)
└── paper/
    └── manuscript_v3.md              # Full English manuscript draft (R2, dyad-level stats filled)
```

### Core Library: `src/lagcc.py` (v1.1)

The **lagged cross-correlation** engine. Key functions:

| Function | Input | Output | Purpose |
|---|---|---|---|
| `lagged_cross_corr(x, y, max_lag)` | 1D arrays | `(lags, corrs, best_lag)` | Lagged CC with normalisation |
| `role_lag(A, B, max_lag)` | 2D `(T, D)` | `int` | Sign of lag → leader identity (A leads if +) |
| `batch_role_lag(pairs)` | list of (A,B) | list of lags | Parallel batch estimation |
| `role_lag_detail(A, B, max_lag)` | 2D `(T, D)` | `(lag, peak_r, consensus, curve, lags)` | Full detail incl. coupling strength & channel consensus |
| `normalize_segment(seg)` | `(T, D, axes)` | normalised | Z-score per channel |
| `velocity_energy(seg)` | `(T, D, 3)` | `(T-1,)` | Frame-wise kinetic energy |

**Design principle**: `role_lag` accepts any 2D input `(T, D)` and does **not** assume a channel ordering — robust to unknown marker/axis majorisation.

### Analysis Scripts

| Script | Experiment | Key Output |
|---|---|---|
| `pilot_lag_bigand.py` | 4-representation comparison + condition statistics | `pilot_lag.csv` |
| `sw_subwindow_lag.py` | Long-window trap validation + sub-window narrowing | `subwindow_lag.csv` |
| `song_influence_lag.py` | Same-song vs different-song coupling (Mann-Whitney, chi-square) | `song_influence.csv` |
| `coherence_analysis.py` | Frequency-band coherence (Friedman, condition × band) | `coherence.csv` |

---

## 4. How to Run

### 4.1 Setup

```bash
# Clone the Bigand data repository
git clone https://github.com/TheoPulse/Dyadic-dance-coordination.git /root/autodl-tmp/Dyadic-dance-coordination

# Place LeadScope code
cd /root/autodl-tmp
# (copy or extract leadscope/ here so that leadscope/src/lagcc.py exists)

# Install deps
pip install numpy scipy joblib torch
```

### 4.2 Run the Full Pipeline

All scripts assume the Bigand repo is at `/root/autodl-tmp/Dyadic-dance-coordination` (override with `--repo`).

```bash
cd /root/autodl-tmp/leadscope

# (1) Pilot lag estimation — 4 representations, full 558 windows
python scripts/pilot_lag_bigand.py --mode energy --max-lag 15 --n-pilot 0

# (2) Sliding-window time-varying lag (50-frame sub-windows)
python scripts/sw_subwindow_lag.py --sub-window 50 --max-lag 5 --mode energy

# (3) Music-symmetry analysis (same-song vs different-song)
python scripts/song_influence_lag.py

# (4) Frequency-band coherence analysis
python scripts/coherence_analysis.py

# (5) Cross-validation on vision-only dataset
python scripts/sw_subwindow_lag.py --data-dir dyadic_dance/data/vis_only --sub-window 50 --max-lag 5 --mode energy
python scripts/coherence_analysis.py --data-dir dyadic_dance/data/vis_only
```

### 4.3 Diagnostic Scripts (optional)

```bash
# Inspect data structure
python scripts/probe_bigand_data.py --repo /root/autodl-tmp/Dyadic-dance-coordination

# Check LFS pointer status (will report 25Hz unavailable)
python scripts/download_lfs_25hz.py --repo /root/autodl-tmp/Dyadic-dance-coordination
```

---

## 5. Output Locations

All results are written to `results/` (auto-created):

| File | Rows | Content |
|---|---|---|
| `pilot_lag.csv` | 558 | Per-window: label, dyad, song_left/right, mode, lag_frames, lag_ms, peak_r, significant |
| `subwindow_lag.csv` | 1674 | Per-sub-segment: window, sub_idx, label, dyad, lag_ms, peak_r |
| `song_influence.csv` | 1674 | Per-sub-segment: window, song_left, song_right, same_song, label, dyad, lag_ms, peak_r |
| `coherence.csv` | 558 | Per-window: label, same_song, low/rhythm/fast band coherence values |

Each script also prints a full statistical report to stdout (redirect with `2>&1 | tee run.log`).

---

## 6. Key Results Summary

**Window-level (exploratory, N=558/1674, not corrected for nesting):**

| Finding | Statistic | Section |
|---|---|---|
| Dyadic coupling structure real | Mismatch-pair Cohen's d = 4.00–6.37 | §4.1 |
| Energy representation optimal | peak_r 0.39 vs 0.08 (5× over raw) | §4.2 |
| Long-window lag scatter trap | 1146 ms → 420 ms (63% narrowing) | §4.3 |
| Frequency-band specific coupling (window-level) | Friedman χ²=44.13, p<0.0001 | §4.5 |

**Dyad-level confirmatory (N=35, review-corrected, FDR-adjusted, all 4/5 survive):**

| Finding | Test | Effect | p (FDR-adj) | Status |
|---|---|---|---|---|
| Same-music > different-music | Mann-Whitney U | **d = 1.128** (corrected from 5.80) | < .0001 | ✓ survives |
| Vision ↑ low-band coherence | Wilcoxon | r_rb = +0.813, d = 1.010 | < .0001 | ✓ survives |
| Vision ↑ rhythm-band coherence | Wilcoxon | r_rb = +0.933, d = 1.038 | < .0001 | ✓ survives |
| Vision ↑ fast-band coherence | Wilcoxon | **r_rb = +0.997, d = 1.646** | < .0001 | ✓ survives (strongest) |
| Lag-direction condition comparison | Wilcoxon | — | non-sig | ✗ fails (5 Hz limit) |

> The original window-level *d* = 5.80 was an SE-in-denominator artefact (per Reviewer 2 audit). The corrected dyad-level *d* = 1.128 remains a large effect; all four confirmatory findings survive FDR correction.

**Figures** (Morandi palette, Times New Roman ≥12pt, no captions):
- `figures/fig1_coherence_spectrum.png` — coherence spectrum with band shading + surrogate null
- `figures/fig2_band_vision.png` — band × vision bar plot with d annotations
- `figures/fig3_lag_sensitivity.png` — sub-window length sensitivity
- `figures/fig4_song_symmetry.png` — same vs different music coupling

---

## 7. Reproducibility Notes

- Random seeds are fixed (`np.random.default_rng(0)` for null distributions) → results are deterministic.
- All statistical tests are non-parametric (Mann-Whitney U, Friedman χ², chi-square) — no normality assumption.
- The `scipy` dependency is optional: scripts degrade to manual normal-approximation Mann-Whitney if `scipy` is absent.
- Compute time: ~2 s per 558-window analysis on 8-core CPU; coherence ~9 s.

---

## 8. Citation

If you use this code, please cite:

```
[Manuscript reference TBD — see paper/manuscript.md]
```

and the original dataset:

```
Bigand, F., et al. (2024). The geometry of interpersonal synchrony in human dance. Current Biology, 34(13), 3011-3019.
```

## 9. License & Contact

Code: MIT License. Data: governed by the Bigand et al. / IIT Dataverse terms.

For questions about the analysis pipeline, refer to `paper/manuscript.md` for the full methodological description.
