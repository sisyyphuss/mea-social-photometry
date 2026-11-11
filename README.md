# Bilateral Medial Amygdala Photometry & Social Behavior Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Computational pipeline for dual-channel fiber photometry, 3-chamber social tracking, and inter-hemispheric spectral coherence in young ($n=8$) versus aged ($n=10$) mice.

---

## Key Results

Analysis across 527 sniffing bouts (381 social, 146 object) reveals a stimulus-selective medial amygdala (MeA) reactivity deficit in aged mice:

| Metric | Young ($n=8$) | Old ($n=10$) | Statistical Test | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Peak MeA Z-score (0 to 4 s)** | $1.585 \pm 0.108$ | $1.103 \pm 0.097$ | Welch $p = 0.0046$ / LMM $p = 0.0003$ | Significant aging deficit |
| **Whole-Bout Mean Z (Duration-Normalized)** | $0.301 \pm 0.076$ | $0.016 \pm 0.086$ | Welch $p = 0.0245$ / LMM $p = 0.0105$ | Significant aging deficit |
| **Fixed 0 to +2 s Mean Z Window** | $0.340 \pm 0.081$ | $0.077 \pm 0.079$ | Welch $p = 0.0334$ / LMM $p = 0.0137$ | Significant aging deficit |
| **Basal Transient Rate (Events/min)** | $2.39 \pm 0.31$ | $1.30 \pm 0.27$ | Welch $p = 0.0170$ | Basal hypoexcitability in aging |
| **Social vs. Object `auROC` Decoding** | $0.591 \pm 0.051$ | $0.419 \pm 0.057$ | Welch $p = 0.0383$ | Loss of social stimulus discrimination |
| **Sustained ($\ge 3\text{ s}$) Bout Peak Z** | $1.763$ | $0.802$ | Welch $p = 0.0018$ | Deficit concentrated in prolonged bouts |
| **Lateralization Index (LI)** | $+0.026 \pm 0.130$ | $-0.120 \pm 0.102$ | Welch $p = 0.3917$ | Symmetric bilateral recruitment |
| **Full Spectrum Coherence (0.1 to 3 Hz)** | $0.596 \pm 0.054$ | $0.590 \pm 0.032$ | Welch $p = 0.3842$ | High baseline synchronization |
| **Locomotor Distance (px)** | $30,850 \pm 5,224$ | $34,788 \pm 5,004$ | Welch $p = 0.5939$ | No motor impairment |

---

## Methods Overview

* **Motion Correction & Optical QC:** Linear regression fitting of 410 nm reference against 470 nm activity isolates calcium signals, removing velocity correlation ($r = 0.11$ raw to $r = 0.01$ corrected, $p = 0.0017$) with equal SNR across cohorts ($p = 0.79$).
* **Low-Pass Filter:** 2nd-order zero-phase Butterworth filter (3.0 Hz cutoff).
* **Baseline & Duration Normalization:** Session Z-scoring followed by pre-stimulus baseline subtraction (-3.0 s to 0.0 s); whole-bout mean Z normalizes response by exact bout duration across 0.3 s to 51.0 s bouts.
* **Mixed-Effects & Non-Parametric Validation:** Trial-level linear mixed models (`~ Age + Bout + Duration + (1 | Mouse_ID)`), Mann-Whitney U, bias-corrected Hedges' $g$, 18-fold leave-one-mouse-out jackknife, and consecutive-threshold PETH windows ($\ge 0.2\text{ s}$).
* **Transients & Selectivity:** MAD-thresholded discrete calcium peak rate outside vs. inside social bouts, single-trial `auROC` social vs. object decoding, duration stratification, bout offset alignment, chamber-entry controls, and lagged cross-correlation.
* **Spectral Coherence & Tracking:** Magnitude-squared Welch coherence (0.1 to 3.0 Hz) and 60 fps 3-chamber spatial occupancy heatmaps with Social Preference Index (SPI).

---

## Repository Structure

```
mea-social-photometry/
├── src/                                  # Core computational modules
│   ├── preprocessing.py                  # Butterworth filter, 410 nm motion correction, dF/F, Z-score
│   ├── peth.py                           # Event bout extraction, PETH slicing, trial metrics, LI
│   ├── coherence.py                      # Multi-band inter-hemispheric spectral coherence
│   ├── tracking.py                       # Coordinates, distance, dwell times, SPI, 2D heatmaps
│   ├── qc.py                             # 410 nm control PETH, bleaching, SNR, velocity scrubbing
│   ├── transients.py                     # MAD-thresholded spontaneous and evoked calcium burst rates
│   ├── modeling.py                       # Linear mixed-effects models, LOMO jackknife, PETH windows
│   ├── behavior_neural.py                # auROC decoding, stratification, offset, zone entry, xcorr
│   └── visualization.py                  # Publication figure styling and palettes
├── run_cohort_analysis.py                # Master cohort pipeline driver
├── run_extended_analysis.py              # Extended QC, mixed modeling, transients, Figures 7-10
├── analyze_mouse.py                      # CLI tool to run any individual animal
├── verify_pipeline.py                    # 8-pass scientific and code verification audit
├── data/
│   └── README.md                         # Raw data format specification
├── results/
│   ├── figures/                          # Publication figures (Figures 1-10 in PNG and PDF)
│   ├── tables/                           # Master datasets, LMM, LOMO, and statistical reports
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

### 2. Run Cohort & Extended Analysis

```bash
python run_cohort_analysis.py
python run_extended_analysis.py
```

### 3. Analyze a Single Animal

```bash
python analyze_mouse.py --mouse_id 303 --age_group Young
```

Or execute any mouse's standalone script directly:
```bash
cd results/individual/Young/Mouse_303
python analyze_Mouse_303.py
```

### 4. Run 8-Pass Verification Audit

```bash
python verify_pipeline.py
```

---

## Publication Figures

* **Figure 1 | Grand Average MeA Dynamics:** Event-aligned PETH comparison for social sniffing and object exploration ([PNG](results/figures/Figure1_Grand_PETH_Dynamics.png) | [PDF](results/figures/Figure1_Grand_PETH_Dynamics.pdf)).
* **Figure 2 | Hemispheric Lateralization Index:** Bilateral symmetry distributions during social sniffing ([PNG](results/figures/Figure2_Hemispheric_Lateralization.png) | [PDF](results/figures/Figure2_Hemispheric_Lateralization.pdf)).
* **Figure 3 | MeA Activation Deficits in Aging:** Peak Z, Whole-Bout Mean Z, and fixed 0 to 2 s window ([PNG](results/figures/Figure3_MeA_Peak_Activation_Aging_Deficit.png) | [PDF](results/figures/Figure3_MeA_Peak_Activation_Aging_Deficit.pdf)).
* **Figure 4 | Behavioral Preference & Locomotor Kinematics:** Social Preference Index (SPI) and distance traveled controls ([PNG](results/figures/Figure4_Behavioral_SPI_and_Locomotion.png) | [PDF](results/figures/Figure4_Behavioral_SPI_and_Locomotion.pdf)).
* **Figure 5 | 2D Spatial Neural Representation:** Spatial coordinate mapping of mean Left and Right MeA Z-scores ([PNG](results/figures/Figure5_Spatial_Neural_Mapping.png) | [PDF](results/figures/Figure5_Spatial_Neural_Mapping.pdf)).
* **Figure 6 | Inter-Hemispheric Bilateral Spectral Coherence:** Coherence spectra across 0.1 to 3.0 Hz and sub-band deltas ([PNG](results/figures/Figure6_Bilateral_Spectral_Coherence_and_Aging.png) | [PDF](results/figures/Figure6_Bilateral_Spectral_Coherence_and_Aging.pdf)).
* **Figure 7 | Optical Quality Controls:** 410 nm isosbestic control PETH, speed-artifact scrubbing, SNR, and object-response positive control ([PNG](results/figures/Figure7_Optical_QC_and_Isosbestic_Controls.png) | [PDF](results/figures/Figure7_Optical_QC_and_Isosbestic_Controls.pdf)).
* **Figure 8 | Statistical Rigor:** Bootstrapped consecutive-threshold PETH windows, duration-bias removal, 18-fold LOMO jackknife, and trial-level LMM ([PNG](results/figures/Figure8_Statistical_Rigor_LMM_Jackknife_and_PETH_Windows.png) | [PDF](results/figures/Figure8_Statistical_Rigor_LMM_Jackknife_and_PETH_Windows.pdf)).
* **Figure 9 | Excitability, Selectivity & Cross-Correlation:** Basal vs. social transient rates, single-trial `auROC`, and lagged bilateral cross-correlation ([PNG](results/figures/Figure9_Spontaneous_Transients_and_Social_Selectivity.png) | [PDF](results/figures/Figure9_Spontaneous_Transients_and_Social_Selectivity.pdf)).
* **Figure 10 | Behavioral Specificity & Coupling:** Duration stratification, zone-entry vs. bout offset, habituation, and brain-behavior decoupling ([PNG](results/figures/Figure10_Bout_Stratification_Offset_ZoneEntry_and_Behavior.png) | [PDF](results/figures/Figure10_Bout_Stratification_Offset_ZoneEntry_and_Behavior.pdf)).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
