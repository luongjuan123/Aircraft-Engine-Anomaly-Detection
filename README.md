# Aircraft Engine Anomaly Detection

Unsupervised Anomaly Detection on Aircraft Turbofan Engine Degradation Datasets using Gaussian Mixture Models (GMM) with Expectation-Maximization (EM).

---

## 📌 Project Overview

This repository develops and evaluates machine learning algorithms for detecting degradation anomalies in aircraft turbofan engines prior to functional failure. The framework uses statistical modeling with a custom, pure NumPy implementation of **Gaussian Mixture Models (GMM)** trained via the **Expectation-Maximization (EM)** algorithm to model normal operational flight regimes and detect degradation through log-likelihood scoring.

### Datasets
1. **Folder 6: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
   - Combined sub-datasets: FD001, FD002, FD003, FD004
   - 24 operational and sensor features across 6 operating regimes
   - 80% Train / 10% Dev / 10% Test split ($N_{total} \approx 265,000$ cycles)
2. **Folder 17: NASA N-CMAPSS (Next-Generation C-MAPSS)**
   - Advanced transient flight data under diverse operating and degradation scenarios

---

## 📁 Repository Structure

```text
Aircraft-Engine-Anomaly-Detection/
├── .gitignore
├── README.md
├── Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/
│   ├── gmm.py                          # Pure NumPy Gaussian Mixture Model with EM
│   ├── preprocess.py                   # Data ingestion, RUL computation, 80/10/10 split
│   ├── train_anomaly_detection.py       # Training, threshold calibration, evaluation & plotting
│   └── plots/                          # Generated visual diagnostics
│       ├── gmm_training_and_evaluation.png
│       └── engine_degradation_trajectories.png
└── Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/
    ├── gmm.py                          # GMM implementation
    ├── preprocess.py                   # N-CMAPSS preprocessing pipeline
    └── train_anomaly_detection.py      # N-CMAPSS training and evaluation
```

> **Note on Data**: Raw and processed datasets are excluded from version control via `.gitignore` due to file size constraints (>28 GB raw, >2.2 GB processed).

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/luongjuan123/Aircraft-Engine-Anomaly-Detection.git
cd Aircraft-Engine-Anomaly-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

### 2. Preprocess Data (Folder 6)
To combine FD001–FD004, compute Remaining Useful Life (RUL), and generate `train.csv`, `dev.csv`, `test.csv`:

```bash
python "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/preprocess.py"
```

### 3. Train & Evaluate Anomaly Detection
To train the GMM ($k=6$), calibrate the detection threshold, and generate performance plots:

```bash
python "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/train_anomaly_detection.py"
```

Command-line arguments:
- `--k`: Number of Gaussian mixture components (default: `6`)
- `--max_iter`: Maximum EM iterations (default: `50`)
- `--reg_cov`: Covariance regularization factor (default: `1e-3`)
- `--no-plot`: Disable generating visual plots
- `--plot-dir`: Directory to save visual plots (default: `plots/`)

---

## 📊 Benchmark Results (Folder 6)

| Metric | Dev Set | Test Set ($N = 26,525$) |
| :--- | :---: | :---: |
| **Log-Likelihood Threshold ($\tau$)** | 53.3649 | 53.3649 |
| **Accuracy** | 61.49% | **61.88%** |
| **Precision** | 56.75% | **56.75%** |
| **Recall (Sensitivity)** | 82.16% | **82.23%** |
| **$F_1$ Score** | 67.13% | **67.15%** |
| **ROC-AUC** | — | **0.7310** |
| **PR-AUC (Average Precision)** | — | **0.7528** |

### Visual Diagnostics
- **Evaluation Dashboard** (`plots/gmm_training_and_evaluation.png`): 6-panel summary covering EM convergence, score distributions, confusion matrix, ROC curve, PR curve, and RUL vs. health score.
- **Engine Degradation Trajectory** (`plots/engine_degradation_trajectories.png`): Lifecycle health tracking for individual turbofans, demonstrating **80–90 cycles of advance warning** before engine failure.