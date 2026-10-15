# Aircraft Engine Anomaly Detection

Unsupervised Anomaly Detection on Aircraft Turbofan Engine Degradation Datasets using Pure NumPy Gaussian Mixture Models (GMM) with Expectation-Maximization (EM).

---

## Project Overview

This repository develops, evaluates, and benchmarks statistical machine learning algorithms for detecting mechanical degradation anomalies in aircraft turbofan engines prior to functional failure. 

The framework employs statistical density estimation with a custom, pure NumPy implementation of **Gaussian Mixture Models (GMM)** trained via the **Expectation-Maximization (EM)** algorithm. By learning the multivariate probability density function of nominal (healthy) flight operations, anomalous operating cycles and progressive component wear are detected through negative log-likelihood anomaly scoring:

$$\text{Anomaly Score}(x) = -\log p(x) = -\log \left( \sum_{j=1}^k \phi_j \mathcal{N}(x; \mu_j, \Sigma_j) \right)$$

When the statistical likelihood of an observed state falls below a calibrated decision boundary $\tau$, the engine is flagged as operating outside its healthy envelope, providing actionable advance warning before mechanical failure occurs.

---

## Datasets

The repository implements and benchmarks anomaly detection across two generations of NASA aircraft gas-turbine degradation benchmarks:

### 1. Folder 6: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Sub-Datasets**: Combined across all four operational variants: **FD001, FD002, FD003, and FD004**.
- **Telemetry Type**: Discrete flight-cycle snapshots taken during steady-state cruise.
- **Dimensionality**: 24 active features (3 operational settings + 21 sensor channels).
- **Regimes**: 6 distinct flight operating regimes dictated by altitude, Mach number, and throttle angle.
- **Dataset Size**: **265,256 total operational flight cycles** partitioned into:
  - 80% Training ($N = 212,206$)
  - 10% Dev / Validation ($N = 26,525$)
  - 10% Held-Out Test ($N = 26,525$)
- **Degradation Label**: Nominal vs. degraded states defined using the standard domain cutoff: cycles with Remaining Useful Life $RUL > 125$ cycles are labeled nominal ($y = 0$), while cycles with $RUL \le 125$ cycles are labeled anomalous ($y = 1$).
- **Status**: **Fully preprocessed, trained, calibrated, and benchmarked.**

### 2. Folder 17: NASA N-CMAPSS (Next-Generation C-MAPSS)
- **Sub-Dataset**: **DS02** (`N-CMAPSS_DS02-006.h5`), modeling real commercial flight profiles.
- **Telemetry Type**: Continuous high-frequency (**1 Hz**) transient flight telemetry covering entire flight profiles: taxi/idle, climb, varied cruise altitudes, and descent/approach.
- **Dimensionality**: 18 active joint features:
  - **4 Scenario Descriptors ($W$)**: Altitude (`alt` [ft]), Flight Mach number (`Mach` [-]), Throttle-resolver angle (`TRA` [%]), and Total inlet temperature (`T2` [°R]).
  - **14 Physical Sensor Measurements ($X_s$)**: Gas path temperatures (`T24`, `T30`, `T48`, `T50`), pressures (`P15`, `P2`, `P21`, `P24`, `Ps30`, `P40`, `P50`), spool speeds (`Nf`, `Nc`), and fuel flow (`Wf`).
- **Fleet Structure**: 9 commercial turbofan units:
  - **Dev Fleet ($N = 5,263,447$ cycles)**: Units 2, 5, 10, 16, 18, 20.
  - **Held-Out Test Fleet ($N = 1,253,743$ cycles)**: Units 11, 14, 15.
- **Degradation Label**: Ground-truth physical health state flag $h_s$:
  - $h_s = 1.0$: Nominal operation before degradation onset ($cycle < t_s$).
  - $h_s = 0.0$: Abnormal degradation regime beginning at transition cycle $t_s$ and propagating to functional failure at end-of-life $t_{EOL}$.
- **Status**: **Fully preprocessed, trained, calibrated, and benchmarked.**

---

## Theoretical Framework

### 1. Gaussian Mixture Model (GMM)
A Gaussian Mixture Model models the probability density of an observation $x \in \mathbb{R}^d$ as a linear superposition of $k$ Gaussian components:

$$p(x; \phi, \mu, \Sigma) = \sum_{j=1}^k \phi_j \mathcal{N}(x; \mu_j, \Sigma_j)$$

where:
- $\phi_j \in [0, 1]$ represents the mixture weight of component $j$, satisfying $\sum_{j=1}^k \phi_j = 1$.
- $\mu_j \in \mathbb{R}^d$ is the mean vector of component $j$.
- $\Sigma_j \in \mathbb{R}^{d \times d}$ is the covariance matrix of component $j$.

The multivariate normal density is evaluated via:

$$\mathcal{N}(x; \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{d/2} |\Sigma_j|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_j)^T \Sigma_j^{-1} (x - \mu_j) \right)$$

### 2. Expectation-Maximization (EM) Algorithm
The model parameters $\theta = \{\phi_j, \mu_j, \Sigma_j\}_{j=1}^k$ are optimized by maximizing the data log-likelihood via EM:

1. **Expectation Step (E-step)**: Compute the posterior responsibilities $w_j^{(i)}$ using numerically stable Log-Sum-Exp:
   $$w_j^{(i)} = P(z^{(i)} = j \mid x^{(i)}; \theta) = \frac{\phi_j \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{l=1}^k \phi_l \mathcal{N}(x^{(i)}; \mu_l, \Sigma_l)}$$

2. **Maximization Step (M-step)**: Update parameters to maximize the expected complete-data log-likelihood:
   $$\phi_j = \frac{1}{m} \sum_{i=1}^m w_j^{(i)}$$
   $$\mu_j = \frac{\sum_{i=1}^m w_j^{(i)} x^{(i)}}{\sum_{i=1}^m w_j^{(i)}}$$
   $$\Sigma_j = \frac{\sum_{i=1}^m w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^m w_j^{(i)}} + \lambda I$$

3. **Covariance Regularization**: Regularization factor $\lambda = 10^{-3}$ is added to the diagonal of $\Sigma_j$ to ensure positive definiteness and prevent singular covariance matrices in lower-variance dimensions.

### 3. Zero-Leakage Standardization
To prevent data leakage between normal and degraded operational regimes, feature normalization parameters $(\mu_{\text{healthy}}, \sigma_{\text{healthy}})$ are calculated **exclusively on healthy training cycles** ($h_s = 1$ / $RUL > 125$):

$$x_{\text{scaled}} = \frac{x - \mu_{\text{healthy}}}{\sigma_{\text{healthy}}}$$

The dev and test sets are transformed using these exact parameters, preserving strict statistical separation.

---

## Repository Structure

```text
Aircraft-Engine-Anomaly-Detection/
├── .gitignore
├── README.md
├── plot_pca_features.py                # Unified PCA feature diagnostics & cross-dataset benchmark runner
├── Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/
│   ├── gmm.py                          # Pure NumPy Gaussian Mixture Model with EM
│   ├── preprocess.py                   # Zero-leakage standardizer & feature scaler
│   ├── train_anomaly_detection.py      # Training, threshold calibration, evaluation & plotting
│   ├── pca_analysis.py                 # PCA feature dimensionality reduction & biplot diagnostics
│   ├── milestone.ipynb                 # Interactive end-to-end analysis and evaluation notebook
│   ├── Data/                           # Raw split CSVs and preprocessed .npy arrays (excluded from git)
│   │   ├── train.csv, dev.csv, test.csv
│   │   ├── X_train_healthy.npy, y_train.npy
│   │   ├── X_dev.npy, y_dev.npy
│   │   ├── X_test.npy, y_test.npy
│   │   └── pipeline.pkl
│   └── Output/                         # Production model artifact and visual diagnostics
│       ├── gmm_model.pkl               # Serialized trained GMM and scaling parameters
│       ├── gmm_training_and_evaluation.png
│       ├── engine_degradation_trajectories.png
│       ├── pca_feature_analysis.png    # 6-panel PCA scree, biplot, regimes & degradation plot
│       └── cross_dataset_pca_comparison.png
└── Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/
    ├── gmm.py                          # Pure NumPy Gaussian Mixture Model with EM
    ├── preprocess.py                   # Zero-leakage standardizer & HDF5 telemetry preprocessor
    ├── train_anomaly_detection.py      # Training, threshold calibration, test evaluation & plotting
    ├── pca_analysis.py                 # PCA feature dimensionality reduction & biplot diagnostics
    ├── milestones.ipynb                # Interactive end-to-end analysis and evaluation notebook
    ├── Data/                           # HDF5 datasets and preprocessed .npy feature arrays (excluded from git)
    │   ├── N-CMAPSS_DS02-006.h5 (and other N-CMAPSS sub-datasets)
    │   ├── X_train_healthy.npy
    │   ├── X_dev.npy, y_dev.npy, A_dev.npy, Y_dev.npy
    │   ├── X_test.npy, y_test.npy, A_test.npy, Y_test.npy
    │   └── pipeline.pkl
    └── Output/                         # Production model artifact and visual diagnostics
        ├── gmm_model.pkl               # Serialized trained GMM and scaling parameters
        ├── gmm_training_and_evaluation.png
        ├── engine_degradation_trajectories.png
        ├── pca_feature_analysis.png    # 6-panel PCA scree, biplot, regimes & degradation plot
        └── cross_dataset_pca_comparison.png
```

> **Note on Data Management**:
> All large raw datasets (`.csv`, `.h5`) and preprocessed feature arrays (`.npy`) reside in the respective `Data/` directories and are excluded from version control via `.gitignore` to maintain repository portability. The `Output/` folders store the serialized model artifacts (`gmm_model.pkl`) and high-resolution diagnostic dashboards.

---

## Getting Started

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/luongjuan123/Aircraft-Engine-Anomaly-Detection.git
cd Aircraft-Engine-Anomaly-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn h5py
```

---

### 2. Reproducing Folder 6 (NASA C-MAPSS)

#### Step 2A: Preprocess and Scale Features
Standardize all active operational and sensor features using statistics fitted exclusively on healthy training cycles ($RUL > 125$):

```bash
python "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/preprocess.py" --method joint
```

Command-line options:
- `--data-dir`: Directory containing `train.csv`, `dev.csv`, `test.csv` (default: `Data/`)
- `--output-dir`: Output directory for `.npy` and `.pkl` files (default: `Data/`)
- `--method`: Feature strategy: `joint` (default benchmark) or `regime_norm`
- `--rul-threshold`: Cutoff for nominal healthy cycles vs. degradation onset (default: `125`)

#### Step 2B: Train GMM and Benchmark Anomaly Detection
Fit GMM ($k=6$), calibrate threshold $\tau$ on Dev, evaluate on held-out Test split, and generate visual figures:

```bash
python "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/train_anomaly_detection.py"
```

Command-line options:
- `--k`: Number of Gaussian mixture components (default: `6`, representing the 6 operating regimes)
- `--max_iter`: Maximum EM iterations (default: `50`)
- `--reg_cov`: Covariance regularization factor (default: `1e-3`)
- `--subsample`: Healthy training cycles subsample size (default: `50000`)
- `--no-plot`: Disable generating diagnostic figures

---

### 3. Reproducing Folder 17 (NASA N-CMAPSS)

#### Step 3A: Preprocess High-Frequency Telemetry
Extract joint 18-dimensional telemetry ($W + X_s$), apply zero-leakage standardization fitted strictly on nominal flight cycles ($h_s = 1$), and export NumPy tensors:

```bash
python "Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/preprocess.py" --method joint
```

Command-line options:
- `--data-file`: Path to HDF5 dataset file (default: `Data/N-CMAPSS_DS02-006.h5`)
- `--output-dir`: Output directory for preprocessed tensors (default: `Data/`)
- `--method`: Feature mode: `joint` (default: 4 flight conditions $W$ + 14 physical sensors $X_s$), `sensors` (14 physical sensors), or `all` (including virtual sensors $X_v$)
- `--subsample-train`: Subsample size of healthy cycles for GMM fitting (default: `50000`)
- `--stride`: Downsampling stride for dev/test arrays (default: `1` for full 1 Hz resolution)

#### Step 3B: Train GMM and Benchmark Anomaly Detection
Fit GMM ($k=6$), calibrate decision threshold $\tau$ on Dev, evaluate across **1,253,743 test cycles**, and generate diagnostic figures:

```bash
python "Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/train_anomaly_detection.py"
```

Command-line options:
- `--k`: Number of Gaussian mixture components (default: `6`, representing multi-regime commercial flight envelopes)
- `--max_iter`: Maximum EM iterations (default: `50`)
- `--reg_cov`: Covariance regularization factor (default: `1e-3`)
- `--subsample`: Healthy training cycles subsample size (default: `50000`)
- `--criterion`: Decision threshold calibration criterion: `balanced_f1` (default), `youden`, or `f1`
- `--no-plot`: Disable generating diagnostic figures

---

## Benchmark Results

### Cross-Dataset Performance Benchmark

The trained GMM models are evaluated on unseen held-out test splits:

| Metric | Folder 6: NASA C-MAPSS | **Folder 17: NASA N-CMAPSS** |
| :--- | :---: | :---: |
| **Telemetry Resolution** | Discrete cycle cruise snapshots | **Continuous 1 Hz transient time-series** |
| **Operational Dynamics** | 6 steady-state flight regimes | **Commercial flight profiles (Climb, Cruise, Descent)** |
| **Target Grounding** | Heuristic threshold ($RUL \le 125$) | **Ground-Truth Physical State ($h_s$)** |
| **Active Feature Space** | 24 features (3 settings, 21 sensors) | **18 features (4 flight conditions, 14 physical sensors)** |
| **Mixture Components ($k$)** | $k=6$ | **$k=6$** |
| **Test Set Size** | 26,525 operational cycles | **1,253,743 operational cycles** |
| **Log-Likelihood Threshold ($\tau$)** | 53.3649 | **32.8624** |
| **Accuracy** | 61.88% | **73.54%** |
| **Precision** | 56.75% | **73.08%** |
| **Recall (Sensitivity)** | 82.23% | **94.33%** |
| **F1 Score** | 67.15% | **82.35%** |
| **ROC-AUC** | 0.7310 | **0.8353** |
| **PR-AUC (Average Precision)** | 0.7528 | **0.9172** |
| **Advance Warning Lead Time** | ~80–90 cycles | **40–43 flight cycles** |

---

### Detailed Test Set Confusion Matrix Breakdown

#### 1. Folder 6 (NASA C-MAPSS, $N = 26,525$ cycles)
- **True Positives (TP)**: 10,337 degraded cycles correctly intercepted (**82.23% recall**)
- **True Negatives (TN)**: 6,076 normal cycles correctly classified
- **False Positives (FP)**: 7,878 nominal cycles flagged (conservative early warning)
- **False Negatives (FN)**: 2,234 degraded cycles missed

#### 2. Folder 17 (NASA N-CMAPSS, $N = 1,253,743$ cycles)
- **True Positives (TP)**: 774,112 degraded cycles correctly intercepted (**94.33% recall**)
- **True Negatives (TN)**: 147,903 normal cycles correctly classified
- **False Positives (FP)**: 285,210 nominal cycles flagged
- **False Negatives (FN)**: 46,518 degraded cycles missed (only **5.67% miss rate**)

---

## Visual Diagnostics: Folder 6 (NASA C-MAPSS)

### Evaluation Dashboard
![GMM Training and Evaluation Dashboard (C-MAPSS)](Work%20on%206.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output/gmm_training_and_evaluation.png)

The C-MAPSS dashboard provides a 6-panel summary:
1. **EM Convergence**: Monotonic increase of log-likelihood across EM iterations, reaching convergence within 9 iterations.
2. **Score Distributions**: Comparison of log-likelihood distributions for normal cycles ($RUL > 125$) and degraded cycles ($RUL \le 125$) against decision boundary $\tau = 53.3649$.
3. **Confusion Matrix Heatmap**: Normalized percentages and raw counts across test predictions.
4. **Receiver Operating Characteristic (ROC)**: ROC curve demonstrating an AUC of **0.7310**.
5. **Precision-Recall Curve**: PR curve demonstrating an Average Precision of **0.7528**.
6. **Health Score vs. Remaining Useful Life**: Binned median trend demonstrating progressive decay in statistical density as engines approach failure.

### Engine Lifecycle Degradation Trajectory
![Engine Degradation Trajectories (C-MAPSS)](Work%20on%206.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output/engine_degradation_trajectories.png)

Condition monitoring across operational lifespans for individual engines (Units #1 and #2 from FD001):
- **Instantaneous vs Rolling Health Index**: 5-cycle rolling smoothing filters sensor noise to yield a stable condition indicator.
- **Advance Warning Lead Time**: A sustained anomaly alert is triggered **80 to 90 flight cycles in advance** of complete engine failure, providing ample operational lead time.

### Principal Component Analysis (PCA) Feature Diagnostics
![C-MAPSS PCA Feature Diagnostics](Work%20on%206.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output/pca_feature_analysis.png)

The C-MAPSS PCA diagnostic dashboard presents a 6-panel decomposition of the 24 operational and sensor channels:
1. **Scree Plot & Cumulative Variance**: PC1 captures **82.26%** and PC2 captures **16.00%**, retaining **98.26% of total operational variance** in a 2D planar subspace (99.51% across top 3 PCs).
2. **PCA Biplot**: Sensor loading vectors show thermodynamic gas-path temperatures and pressures (`s_4`, `s_9`, `s_11`, `s_17`) aligning with PC1, while spool speed corrections (`s_13`, `s_19`) and flight settings (`setting_1`, `setting_3`) align with PC2.
3. **Latent Health Separation**: Clear envelope expansion and dispersion between nominal cruise cycles ($RUL > 125$) and degrading cycles ($RUL \le 125$).
4. **Multi-Regime Clustering**: The 6 operational flight regimes naturally segregate into 6 distinct clusters across the latent plane.
5. **Feature Loadings Distribution**: Quantitative eigenvector weights across all 24 channels.
6. **Engine Lifecycle Trajectory**: Tracking Engine #1 from FD001 across its complete lifespan (Cycle 1 to EOL failure) drifting outward along the degradation manifold.

---

## Visual Diagnostics: Folder 17 (NASA N-CMAPSS)

### Evaluation Dashboard
![N-CMAPSS GMM Training and Evaluation Dashboard](Work%20on%2017.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Output/gmm_training_and_evaluation.png)

The N-CMAPSS dashboard provides a 6-panel summary:
1. **EM Convergence**: Monotonic log-likelihood ascent reaching optimal parameters within 50 iterations.
2. **Score Distributions**: Distinct bimodal separation between nominal flight cycles ($h_s = 1$, centered near $\log p(x) \approx 32$) and degraded cycles ($h_s = 0$, extending down to $\log p(x) \approx -170$).
3. **Confusion Matrix Heatmap**: Full counts and percentages across 1.25M test cycles.
4. **Receiver Operating Characteristic (ROC)**: High discriminative performance with ROC-AUC of **0.8353**.
5. **Precision-Recall Curve**: Robust performance across thresholds with Average Precision of **0.9172**.
6. **Health Score vs. Remaining Useful Life**: Progressive decay in log-likelihood as engines approach end-of-life breakdown ($RUL \to 0$).

### Engine Lifecycle Degradation Trajectories & Advance Warning Lead Times
![N-CMAPSS Engine Degradation Trajectories](Work%20on%2017.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Output/engine_degradation_trajectories.png)

Condition monitoring across complete operational lifespans for all held-out test turbofans:
- **Test Unit #11** (Lifespan: 59 cycles | Fault Onset $t_s = 19$): Sustained alarm triggered at **Cycle 19**, providing **40 flight cycles of advance warning** before breakdown.
- **Test Unit #14** (Lifespan: 76 cycles | Fault Onset $t_s = 36$): Sustained alarm triggered at **Cycle 36**, providing **40 flight cycles of advance warning** before breakdown.
- **Test Unit #15** (Lifespan: 67 cycles | Fault Onset $t_s = 24$): Sustained alarm triggered at **Cycle 24**, providing **43 flight cycles of advance warning** before breakdown.

### Principal Component Analysis (PCA) Feature Diagnostics
![N-CMAPSS PCA Feature Diagnostics](Work%20on%2017.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Output/pca_feature_analysis.png)

The N-CMAPSS PCA diagnostic dashboard presents a 6-panel decomposition of the 18 joint operational and sensor channels:
1. **Scree Plot & Cumulative Variance**: PC1 captures **68.35%** and PC2 captures **30.04%**, retaining **98.39% of total variance** from continuous transient flight missions (99.62% across top 3 PCs).
2. **PCA Biplot**: Contrasts flight scenario descriptors ($W$: `alt`, `Mach`, `TRA`, `T2`) with internal physical sensor measurements ($X_s$). Spool speeds (`Nf`, `Nc`) and `TRA` dominate PC2, while gas path pressures (`P40`, `Ps30`, `P24`) and fuel flow (`Wf`) dominate PC1.
3. **Latent Health Separation**: Ground-truth nominal flight telemetry ($h_s = 1.0$) vs abnormal degradation regime ($h_s = 0.0$).
4. **Commercial Flight Class Manifolds**: Separation across Flight Classes 1, 2, and 3 (short, medium, and long-haul profiles).
5. **Feature Loadings Distribution**: Quantitative eigenvector weights across all 18 channels.
6. **Test Fleet Degradation Trajectories**: Multi-cycle trajectories for Units #11, #14, and #15 drifting away from nominal flight manifold past fault onset $t_s$.

---

## Cross-Dataset PCA Comparative Benchmark

![Cross-Dataset PCA Comparison](Work%20on%2017.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Output/cross_dataset_pca_comparison.png)

Direct visual benchmark comparing dimensionality reduction efficacy, scree decay, and latent anomaly separation between discrete steady-state cruise telemetry (C-MAPSS) and continuous transient flight telemetry (N-CMAPSS):
- **Dimensionality Reduction Efficacy**: In both benchmark generations, the top 2 principal components capture over **98.2% to 98.4% of total variance**, showing that complex high-dimensional turbofan dynamics can be projected into 2D planar space without substantial information loss.
- **Manifold Separation**: Both datasets exhibit prominent divergence between nominal envelopes and degradation states in latent space, providing geometric grounding for the high sensitivity of Gaussian Mixture density estimation.

---

## Key Technical Insights & Industrial Relevance

1. **Continuous Telemetry vs. Discrete Snapshots**:
   High-frequency 1 Hz telemetry in N-CMAPSS captures the dynamic interaction between control inputs (`TRA`), ambient flight conditions (`alt`, `Mach`, `T2`), and gas path thermodynamics. Modeling this joint distribution enhances anomaly detection F1 score from **67.15% to 82.35%**.

2. **Ground-Truth Physical State Grounding**:
   Grounding anomaly detection on the physical health state flag $h_s$ eliminates the subjectivity of arbitrary RUL cutoffs, achieving **94.33% detection sensitivity** (Recall) on unseen turbofans.

3. **Actionable Predictive Maintenance Lead Times**:
   The statistical health index consistently alerts flight operations **40 to 43 cycles in advance** of complete engine breakdown on N-CMAPSS, providing reliable, actionable operational margins for airline fleet maintenance scheduling.

---

## Authors & Citation

**Author**: Lương Chí Dũng ([@luongjuan123](https://github.com/luongjuan123))  
**Repository**: [Aircraft-Engine-Anomaly-Detection](https://github.com/luongjuan123/Aircraft-Engine-Anomaly-Detection)  
**Dataset Attributions**: NASA Prognostics Center of Excellence (PCoE) C-MAPSS and N-CMAPSS Datasets.