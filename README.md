# Bilateral Medial Amygdala Photometry & Social Behavior Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Computational pipeline for dual-channel fiber photometry, 3-chamber social tracking, and inter-hemispheric spectral coherence in young ($n=8$) versus aged ($n=10$) mice.

---

## Key Results

Analysis across 527 sniffing bouts reveals a significant medial amygdala (MeA) reactivity deficit in aged mice during social interaction:

| Metric | Young ($n=8$) | Old ($n=10$) | Welch's $t$ | $p$-value | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Peak MeA Z-score (0 to 4 s)** | $1.585 \pm 0.108$ | $1.103 \pm 0.097$ | $t = 3.326$ | **$p = 0.0046$** | Significant aging deficit |
| **Whole-Bout Mean Z (Duration-Normalized)** | $0.301 \pm 0.076$ | $0.016 \pm 0.086$ | $t = 2.483$ | **$p = 0.0245$** | Significant aging deficit |
| **Fixed 0 to +2 s Mean Z Window** | $0.340 \pm 0.081$ | $0.077 \pm 0.079$ | $t = 2.332$ | **$p = 0.0334$** | Significant aging deficit |
| **Lateralization Index (LI)** | $+0.026 \pm 0.130$ | $-0.120 \pm 0.102$ | $t = 0.884$ | $p = 0.3917$ | Symmetric bilateral recruitment |
| **Full Spectrum Coherence (0.1 to 3 Hz)** | $0.596 \pm 0.054$ | $0.590 \pm 0.032$ | $t = 0.896$ | $p = 0.3842$ | High baseline synchronization |
| **Locomotor Distance (px)** | $30,850 \pm 5,224$ | $34,788 \pm 5,004$ | $t = -0.544$ | $p = 0.5939$ | No motor impairment |

---

## Methods Overview

* **Motion Correction:** Linear regression fitting of 410 nm isosbestic reference to 470 nm signal to isolate calcium-dependent fluorescence.
* **Low-Pass Filter:** 2nd-order zero-phase Butterworth filter (3.0 Hz cutoff).
* **Baseline Normalization:** Session-level Z-scoring, followed by pre-stimulus baseline subtraction (-3.0 s to 0.0 s) for each trial.
* **Duration Normalization:** Whole-bout mean Z computes average baseline-subtracted activation over each bout's exact length, removing duration bias across interactions ranging from 0.3 s to 51.0 s.
* **Spectral Coherence:** Magnitude-squared coherence $C_{xy}(f)$ between Left and Right MeA during pre-stimulus baseline versus interaction across 0.1 to 3.0 Hz and sub-bands (0.1-0.5 Hz, 0.5-1.5 Hz, 1.5-3.0 Hz, 0.8-1.6 Hz).
* **Behavioral Tracking:** 60 fps trajectory tracking across 3 chambers, chamber dwell times, Social Preference Index (SPI), and 2D spatial presence heatmaps.

---

## Repository Structure

```
mea-social-photometry/
├── src/                                  # Core computational modules
│   ├── preprocessing.py                  # Butterworth filter, 410 nm motion correction, dF/F, Z-score
│   ├── peth.py                           # Event bout extraction, PETH slicing, trial metrics, LI
│   ├── coherence.py                      # Multi-band inter-hemispheric spectral coherence
│   ├── tracking.py                       # Coordinates, distance, dwell times, SPI, 2D heatmaps
│   └── visualization.py                  # Publication figure styling and palettes
├── run_cohort_analysis.py                # Master cohort pipeline driver
├── analyze_mouse.py                      # CLI tool to run any individual animal
├── verify_pipeline.py                    # 7-pass scientific and code verification audit
├── data/
│   └── README.md                         # Raw data format specification
├── results/
│   ├── figures/                          # Publication figures (Figures 1-6 in PNG and PDF)
│   ├── tables/                           # Master trial dataset, per-mouse summary, stats report
│   ├── cohorts/                          # Young and Old cohort summary plots and tables
│   └── individual/                       # 18 individual animal folders (8 files per animal)
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md
```

---

## Installation & Usage

### 1. Installation

```bash
git clone https://github.com/sisyyphuss/mea-social-photometry.git
cd mea-social-photometry
pip install -r requirements.txt
```

### 2. Run Cohort Analysis

Processes all 18 animals and regenerates cohort tables and figures:

```bash
python run_cohort_analysis.py
```

Optional arguments:
```bash
python run_cohort_analysis.py --raw_zip path/to/Data_2.zip --tracking_dir path/to/results_social
```

### 3. Analyze a Single Animal

```bash
python analyze_mouse.py --mouse_id 303 --age_group Young
python analyze_mouse.py --mouse_id 315 --age_group Old
```

Or execute any mouse's standalone script directly:
```bash
cd results/individual/Young/Mouse_303
python analyze_Mouse_303.py
```

### 4. Run Verification Audit

Runs an automated 7-pass verification check verifying file tree completeness, zero NaNs, JSON-to-CSV match, statistical replication, and script syntax:

```bash
python verify_pipeline.py
```

---

## Publication Figures

* **Figure 1 | Grand Average MeA Dynamics:** Event-aligned PETH comparison for social sniffing and object exploration ([PNG](results/figures/Figure1_Grand_PETH_Dynamics.png) | [PDF](results/figures/Figure1_Grand_PETH_Dynamics.pdf)).
* **Figure 2 | Hemispheric Lateralization Index:** Bilateral symmetry distributions during social sniffing ([PNG](results/figures/Figure2_Hemispheric_Lateralization.png) | [PDF](results/figures/Figure2_Hemispheric_Lateralization.pdf)).
* **Figure 3 | MeA Activation Deficits in Aging:** 3-panel figure comparing Peak Z, Whole-Bout Mean Z, and fixed 0 to 2 s window ([PNG](results/figures/Figure3_MeA_Peak_Activation_Aging_Deficit.png) | [PDF](results/figures/Figure3_MeA_Peak_Activation_Aging_Deficit.pdf)).
* **Figure 4 | Behavioral Preference & Locomotor Kinematics:** Social Preference Index (SPI) and distance traveled controls ([PNG](results/figures/Figure4_Behavioral_SPI_and_Locomotion.png) | [PDF](results/figures/Figure4_Behavioral_SPI_and_Locomotion.pdf)).
* **Figure 5 | 2D Spatial Neural Representation:** Spatial coordinate mapping of mean Left and Right MeA Z-scores ([PNG](results/figures/Figure5_Spatial_Neural_Mapping.png) | [PDF](results/figures/Figure5_Spatial_Neural_Mapping.pdf)).
* **Figure 6 | Inter-Hemispheric Bilateral Spectral Coherence:** Coherence spectra across 0.1 to 3.0 Hz and sub-band coupling deltas ([PNG](results/figures/Figure6_Bilateral_Spectral_Coherence_and_Aging.png) | [PDF](results/figures/Figure6_Bilateral_Spectral_Coherence_and_Aging.pdf)).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
