# Bilateral Medial Amygdala Photometry & Social Behavior Analysis

Computational pipeline for dual-channel fiber photometry, 3-chamber social tracking, and inter-hemispheric spectral coherence in young ($n = 8$) versus aged ($n = 10$) mice.

---

## Key results

Analysis across 381 sniffing bouts (235 social, 146 object; durations 0.30 s to 50.98 s, mean 5.60 s) shows an evoked medial amygdala (MeA) reactivity deficit and blunted inter-hemispheric coherence recruitment in aged mice despite unchanged locomotion:

| Metric | Young ($n = 8$) | Old ($n = 10$) | Statistical Test | Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Peak MeA Z-score (0 to 4 s)** | $1.772 \pm 0.243$ | $0.978 \pm 0.114$ | Welch $p = 0.0143$ / MW $p = 0.0155$ / LMM $p = 0.0010$ | Blunted evoked response in aging ($g = 1.430$, 18/18 LOMO) |
| **Whole-Bout Mean Z (Duration-Normalized)** | $0.406 \pm 0.145$ | $-0.048 \pm 0.093$ | Welch $p = 0.0211$ / MW $p = 0.0117$ / LMM $p = 0.0032$ | Activation deficit remains after duration normalization ($g = 1.238$) |
| **Fixed 0 to +2 s Mean Z Window** | $0.469 \pm 0.157$ | $0.024 \pm 0.091$ | Welch $p = 0.0312$ / MW $p = 0.0266$ / LMM $p = 0.0045$ | Early sensory trigger deficit ($g = 1.163$, 11/18 LOMO) |
| **Sustained ($\ge 3\text{ s}$) Bout Peak Z** | $1.833$ | $0.828$ | Welch $p = 0.0074$ | Deficit widens during sustained social contact |
| **Short ($< 3\text{ s}$) Bout Peak Z** | $1.707$ | $1.035$ | Welch $p = 0.0396$ | Deficit also present in brief bouts |
| **Basal Transient Rate (Outside Bouts, /min)** | $2.50 \pm 0.27$ | $1.26 \pm 0.24$ | Welch $p = 0.0037$ | Spontaneous basal hypoexcitability in aged mice |
| **Bout Offset Disengagement Delta Z** | $-0.289$ | $+0.056$ | Welch $p = 0.0014$ | Fast neural reset upon disengagement lost in aged mice |
| **Social vs Object `auROC` Decoding** | $0.640 \pm 0.085$ | $0.398 \pm 0.075$ | Welch $p = 0.0505$ / MW $p = 0.0409$ | Loss of single-trial social vs object selectivity |
| **Lateralization Index (LI)** | $-0.151 \pm 0.122$ | $-0.066 \pm 0.161$ | Welch $p = 0.6821$ | Symmetric bilateral recruitment across both cohorts |
| **Full Spectrum Coherence (0.1 to 3.0 Hz)** | $0.602 \pm 0.054$ | $0.581 \pm 0.040$ | Welch delta $p = 0.0099$ (0.8 to 1.6 Hz $p = 0.0464$) | Aged mice fail to recruit task-evoked bilateral coherence |
| **Behavioral Social Preference Index (SPI)** | $0.122 \pm 0.240$ | $0.329 \pm 0.195$ | Welch $p = 0.5135$ | Intact chamber preference in both cohorts |
| **Locomotor Distance (px)** | $30,850 \pm 5,224$ | $34,788 \pm 5,004$ | Welch $p = 0.5939$ | Locomotion is matched between cohorts |

---

## Methods summary

* **Motion correction & optical quality:** Linear regression of 410 nm isosbestic control onto 470 nm fluorescence isolates calcium signals, removing velocity correlation (Young $r = 0.093$ to $0.0002$; Old $r = 0.118$ to $0.0111$; paired drop $p = 0.0020$) with matched session SNR ($32.62 \pm 4.08$ vs $31.09 \pm 4.00$, $p = 0.7927$) and flat 410 nm PETH deflection (Young $-0.0809$, Old $-0.0086$).
* **Filtering & resampling:** Second-order zero-phase Butterworth low-pass filter (3.0 Hz cutoff) with uniform interpolation for exact time alignment at $t = 0.000\text{ s}$.
* **Bout segmentation & duration normalization:** 235 social sniffing bouts and 146 novel object bouts were extracted from behavioral tracking. Continuous trapezoidal integration normalizes whole-bout mean Z by duration ((1/T) * integral(Z dt)), removing correlation with bout length (Young $r = 0.003, p = 0.9769$ vs raw AUC $r = 0.161, p = 0.1077$).
* **Mixed-effects, jackknife & permutation testing:** Trial-level linear mixed models (`~ Age + Bout + Duration + (1 | Mouse_ID)` across 235 social trials), Mann-Whitney U, bias-corrected Hedges' $g$, 18-fold leave-one-mouse-out (LOMO) jackknife, and consecutive-threshold PETH permutation windows ($\ge 0.2\text{ s}$ at $p < 0.05$: `-0.23 to +0.03 s`, `+0.53 to +0.83 s`, and `+1.43 to +2.50 s`).
* **Transients, selectivity & behavior:** MAD-thresholded discrete calcium burst rates (total, basal outside bouts, and social-evoked), single-trial `auROC` discrimination, duration stratification ($< 3\text{ s}$ vs $\ge 3\text{ s}$), bout offset disengagement, chamber entry controls ($p = 0.4733$), and zero-lag bilateral cross-correlation ($p = 0.362$).

---

## Repository structure

```
mea-social-photometry/
├── src/                         # Core signal processing, tracking, coherence, and modeling modules
├── run_all.py                   # Master script to run full pipeline
├── run_cohort_analysis.py       # Primary cohort analysis and Figures 1 to 6
├── run_extended_analysis.py     # Extended statistics, QC, and Figures 7 to 10
├── analyze_mouse.py             # Single animal diagnostic CLI
├── data/                        # Raw photometry & tracking layout instructions (see data/README.md)
├── results/
│   ├── figures/                 # Figures 1 to 10 in PNG (300 dpi) and vector PDF
│   ├── tables/                  # Master datasets, LMM, LOMO, and summary reports
│   ├── cohorts/                 # Young and Old cohort summary plots and tables
│   └── individual/              # Individual animal diagnostic folders (6 files per animal)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
```

---

## Setup and usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/sisyyphuss/mea-social-photometry.git
cd mea-social-photometry
pip install -r requirements.txt
```

Place `Data_2.zip` and the `results_social` tracking folder inside `data/` (or keep them in your `Downloads` directory), then run the analysis:

```bash
python run_all.py
```

To specify custom data paths or run individual parts separately:

```bash
# Custom file locations
python run_all.py --raw_zip path/to/Data_2.zip --tracking_dir path/to/results_social

# Run cohort analysis or extended analysis individually
python run_cohort_analysis.py
python run_extended_analysis.py

# Analyze a single mouse
python analyze_mouse.py --mouse_id 303 --age_group Young
```

---

## Figures

* **Figure 1 | Grand Average MeA Dynamics:** Event-aligned PETH comparison for social sniffing and object exploration ([PNG](results/figures/Figure1_Grand_PETH_Dynamics.png) | [PDF](results/figures/Figure1_Grand_PETH_Dynamics.pdf)).
* **Figure 2 | Hemispheric Lateralization Index:** Bilateral symmetry distributions during social sniffing ([PNG](results/figures/Figure2_Hemispheric_Lateralization.png) | [PDF](results/figures/Figure2_Hemispheric_Lateralization.pdf)).
* **Figure 3 | MeA Activation Deficits in Aging:** Peak Z, Whole-Bout Mean Z, and fixed 0 to 2 s window ([PNG](results/figures/Figure3_MeA_Peak_Activation_Aging_Deficit.png) | [PDF](results/figures/Figure3_MeA_Peak_Activation_Aging_Deficit.pdf)).
* **Figure 4 | Behavioral Preference & Locomotor Kinematics:** Social Preference Index (SPI) and distance traveled controls ([PNG](results/figures/Figure4_Behavioral_SPI_and_Locomotion.png) | [PDF](results/figures/Figure4_Behavioral_SPI_and_Locomotion.pdf)).
* **Figure 5 | 2D Spatial Neural Representation:** Spatial coordinate mapping of mean Left and Right MeA Z-scores ([PNG](results/figures/Figure5_Spatial_Neural_Mapping.png) | [PDF](results/figures/Figure5_Spatial_Neural_Mapping.pdf)).
* **Figure 6 | Inter-Hemispheric Bilateral Spectral Coherence:** Coherence spectra across 0.1 to 3.0 Hz and sub-band deltas ([PNG](results/figures/Figure6_Bilateral_Spectral_Coherence_and_Aging.png) | [PDF](results/figures/Figure6_Bilateral_Spectral_Coherence_and_Aging.pdf)).
* **Figure 7 | Optical Controls:** 410 nm isosbestic control PETH, motion-artifact speed correlation, SNR, and object-response positive control ([PNG](results/figures/Figure7_Optical_QC_and_Isosbestic_Controls.png) | [PDF](results/figures/Figure7_Optical_QC_and_Isosbestic_Controls.pdf)).
* **Figure 8 | Mixed-Effects Modeling & Time Windows:** Bootstrapped consecutive-threshold PETH windows, duration normalization, 18-fold LOMO jackknife, and trial-level LMM ([PNG](results/figures/Figure8_Mixed_Effects_Jackknife_and_PETH_Windows.png) | [PDF](results/figures/Figure8_Mixed_Effects_Jackknife_and_PETH_Windows.pdf)).
* **Figure 9 | Excitability, Selectivity & Cross-Correlation:** Basal vs social transient rates, single-trial `auROC`, and lagged bilateral cross-correlation ([PNG](results/figures/Figure9_Spontaneous_Transients_and_Social_Selectivity.png) | [PDF](results/figures/Figure9_Spontaneous_Transients_and_Social_Selectivity.pdf)).
* **Figure 10 | Behavioral Specificity & Coupling:** Duration stratification, zone-entry vs bout offset, habituation, and brain-behavior coupling ([PNG](results/figures/Figure10_Bout_Stratification_Offset_ZoneEntry_and_Behavior.png) | [PDF](results/figures/Figure10_Bout_Stratification_Offset_ZoneEntry_and_Behavior.pdf)).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
