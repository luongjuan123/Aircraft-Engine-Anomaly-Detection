# Aircraft Engine Anomaly Detection

Unsupervised Anomaly Detection on Aircraft Turbofan Engine Degradation Datasets using Gaussian Mixture Models (GMM) with Expectation-Maximization (EM).

---

## Project Overview

This repository develops and evaluates statistical machine learning algorithms for detecting mechanical degradation anomalies in aircraft turbofan engines prior to functional failure. The framework uses statistical density estimation with a custom, pure NumPy implementation of Gaussian Mixture Models (GMM) trained via the Expectation-Maximization (EM) algorithm to model nominal operational flight envelopes and detect progressive degradation through log-likelihood anomaly scoring.

### Datasets

1. **Folder 6: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
   - Combined sub-datasets: FD001, FD002, FD003, FD004.
   - 24 operational and sensor features across 6 flight operating regimes.
   - 80% Train / 10% Dev / 10% Test split (265,256 total operational flight cycles).
   - Status: Fully preprocessed, trained, calibrated, and benchmarked.

2. **Folder 17: NASA N-CMAPSS (Next-Generation C-MAPSS)**
   - Advanced real flight transient data under diverse flight profiles and degradation scenarios.
   - Status: Still in progress.

---

## Repository Structure

```text
Aircraft-Engine-Anomaly-Detection/
├── .gitignore
├── README.md
├── Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/
│   ├── gmm.py                          # Pure NumPy Gaussian Mixture Model with EM
│   ├── preprocess.py                   # Feature selection, zero-leakage standardizer, and scaling
│   ├── train_anomaly_detection.py      # Training, dev threshold calibration, test evaluation & plotting
│   ├── milestone.ipynb                 # Interactive end-to-end analysis and evaluation notebook
│   ├── Data/                           # Raw split CSVs and preprocessed .npy feature arrays (excluded from git)
│   │   ├── train.csv, dev.csv, test.csv
│   │   ├── X_train_healthy.npy, y_train.npy
│   │   ├── X_dev.npy, y_dev.npy
│   │   ├── X_test.npy, y_test.npy
│   │   └── pipeline.pkl
│   └── Output/                         # Model artifact and visual plots
│       ├── gmm_model.pkl               # Serialized trained GMM and scaling parameters
│       ├── gmm_training_and_evaluation.png
│       └── engine_degradation_trajectories.png
└── Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/
    ├── gmm.py                          # NumPy GMM implementation
    └── milestones.ipynb                # N-CMAPSS exploratory and evaluation notebook (In Progress)
```

Note on Data Management:
All raw datasets and preprocessed feature arrays (`.npy`, `.csv`, `.h5`) reside in the `Data/` folders and are excluded from version control via `.gitignore` to preserve repository portability. The `Output/` folder is reserved exclusively for the trained model artifact (`gmm_model.pkl`) and visual diagnostic figures.

---

## Getting Started

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/luongjuan123/Aircraft-Engine-Anomaly-Detection.git
cd Aircraft-Engine-Anomaly-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Preprocess Data (Folder 6)

Preprocess the combined C-MAPSS dataset, perform zero-leakage standard scaling on healthy operational cycles (RUL > 125 cycles), and save ready-to-train NumPy arrays into `Data/`:

```bash
python "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/preprocess.py" --method joint
```

Command-line arguments:
- `--data-dir`: Directory containing `train.csv`, `dev.csv`, `test.csv` (default: `Data/`)
- `--output-dir`: Target directory for preprocessed arrays (default: `Data/`)
- `--method`: Preprocessing strategy: `joint` (default benchmark) or `regime_norm`
- `--rul-threshold`: Cutoff for nominal healthy cycles vs degradation onset (default: `125`)

### 3. Train and Evaluate Anomaly Detection

Train the GMM on the healthy operational flight envelope, calibrate the decision threshold on the Dev set, evaluate on the Test set, and generate visual figures in `Output/`:

```bash
python "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/train_anomaly_detection.py"
```

Command-line arguments:
- `--k`: Number of Gaussian mixture components (default: `6`, representing the 6 flight operating regimes)
- `--max_iter`: Maximum EM iterations (default: `50`)
- `--reg_cov`: Covariance diagonal regularization factor (default: `1e-3`)
- `--subsample`: Subsample size of healthy cycles for fast EM training (default: `50000`)
- `--no-plot`: Disable generating diagnostic plots

---

## Benchmark Results (Folder 6)

The model is trained strictly on nominal cycles and evaluated on the held-out test split of 26,525 unseen operational cycles across all four C-MAPSS sub-datasets:

| Metric | Dev Set | Test Set (N = 26,525) |
| :--- | :---: | :---: |
| **Log-Likelihood Threshold ($\tau$)** | 53.3649 | 53.3649 |
| **Accuracy** | 61.49% | **61.88%** |
| **Precision** | 56.75% | **56.75%** |
| **Recall (Sensitivity)** | 82.16% | **82.23%** |
| **F1 Score** | 67.13% | **67.15%** |
| **ROC-AUC** | — | **0.7310** |
| **PR-AUC (Average Precision)** | — | **0.7528** |

### Test Set Confusion Matrix Breakdown
- True Positives (TP): 10,337 degraded cycles correctly flagged
- True Negatives (TN): 6,076 normal cycles correctly classified
- False Positives (FP): 7,878 nominal cycles flagged (conservative early warning)
- False Negatives (FN): 2,234 degraded cycles missed

---

## Visual Diagnostics

### Evaluation Dashboard

![GMM Training and Evaluation Dashboard](Work%20on%206.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output/gmm_training_and_evaluation.png)

The evaluation dashboard provides a 6-panel summary:
1. **EM Convergence**: Monotonic increase of log-likelihood across EM iterations, reaching convergence within 9 iterations.
2. **Score Distributions**: Comparison of log-likelihood distributions for normal cycles (RUL > 125) and degraded cycles (RUL <= 125) against the calibrated decision boundary $\tau = 53.3649$.
3. **Confusion Matrix Heatmap**: Normalized percentages and raw counts across test predictions.
4. **Receiver Operating Characteristic (ROC)**: ROC curve demonstrating an AUC of 0.7310.
5. **Precision-Recall Curve**: PR curve demonstrating an Average Precision of 0.7528.
6. **Health Score vs. Remaining Useful Life**: Binned median trend showing progressive decay in log-likelihood as engines approach functional failure.

### Engine Lifecycle Degradation Trajectory

![Engine Degradation Trajectories](Work%20on%206.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output/engine_degradation_trajectories.png)

Condition monitoring across full operational lifespans for individual turbofans (Units #1 and #2 from FD001):
- **Instantaneous vs Rolling Health Index**: 5-cycle rolling smoothing filters high-frequency sensor noise to generate a stable condition indicator.
- **Advance Warning Lead Time**: A sustained anomaly alert is triggered **80 to 90 flight cycles in advance** of complete engine breakdown, providing ample lead time for predictive maintenance intervention.

---

## Work on Dataset 17 (N-CMAPSS)

Investigation on the Next-Generation C-MAPSS dataset (Folder 17) is currently **in progress**. This work addresses complex commercial flight missions with transient flight dynamics across varied climb, cruise, and descent phases using higher-frequency sensor telemetry.