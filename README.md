# AI-Driven Insider Threat Detection Framework for Hybrid Work Environments

This repository contains the implementation for a final year research project focused on detecting insider threats in hybrid work environments using behavioral telemetry from email activity and login/device events.

The current system supports GRU-based sequential anomaly detection, feature engineering, Streamlit dashboards for monitoring and investigation, configurable risk scoring, and JSON report export with saved model artifacts.

## 1. Project Overview

Insider threats are particularly challenging in hybrid work settings due to variable work patterns, remote access, and increased reliance on digital collaboration. This project implements an AI-assisted framework that:

- Preprocesses and engineers features from timestamped event data.
- Builds sequential representations of user behavior (event sequences).
- Detects suspicious behavior using a GRU-based model.
- Surfaces results through Streamlit dashboards.
- Exports machine-readable reports for auditability and downstream analysis.

## 2. Research Objective

The research objective is to design and evaluate a practical insider threat detection framework that combines:

- Sequence-based learning for behavioral signals (current: GRU).
- Interpretable, configurable risk scoring rules for operational tuning.
- Dashboard-based monitoring for analysts.
- Exportable and verifiable alert artifacts (reporting; integrity verification is planned).

## 3. Key Features

Implemented (current):

- **GRU-based sequential anomaly detection** for timestamped event sequences.
- **Feature engineering + preprocessing** aligned with the training pipeline.
- **Streamlit dashboards**:
  - Email Privilege Misuse Detector
  - Login/Device anomaly dashboard
  - Training dashboard and SOC-style overview
- **Configurable risk scoring** (Model / Manual / Hybrid) with rule-by-rule explanation.
- **JSON report export** (download + optional save to disk).
- **Saved trained artifacts** (model weights + preprocessor + metadata).

Planned extensions (in scope for future work):

- Transformer-based sequence model.
- Confidence scoring and calibration.
- Real-time streaming simulation for near-real-time alerting.
- Alert integrity verification using hashing.
- Baseline comparisons against simpler models.

## 4. System Architecture

At a high level, the pipeline is:

1) **Ingest**: CSV event logs (email / device-login)
2) **Preprocess**: parsing, timestamp normalization, feature engineering
3) **Sequence building**: group events (e.g., by user) into fixed-length sequences
4) **Model scoring**: GRU outputs a probability of risky behavior
5) **Risk scoring engine**: model-only / manual rules / hybrid blending
6) **Dashboards + reporting**: visualization, explanations, JSON exports

## 5. Project Structure

Clean tree (key folders):

```text
.
├─ apps/
│  ├─ streamlit_app.py
│  ├─ email_privilege_dashboard.py
│  ├─ login_time_window_dashboard.py
│  ├─ soc_cyber_dashboard.py
│  ├─ training_dashboard.py
│  └─ run_email_dashboard.py
├─ scripts/
│  ├─ train_gru.py
│  ├─ train_login_detector.py
│  ├─ build_labels_from_rules.py
│  ├─ build_labels_from_users.py
│  ├─ build_device_labels_from_time_window.py
│  └─ make_label_mapping_template.py
├─ src/
│  └─ insider_gru/
│     ├─ config.py
│     ├─ data.py
│     ├─ model.py
│     ├─ train.py
│     └─ plots.py
├─ Dataset/
│  ├─ train/
│  └─ test/
└─ outputs/
   ├─ model/
   ├─ reports/
   ├─ plots/
   ├─ login/
   │  ├─ model/
   │  ├─ reports/
   │  └─ plots/
   └─ integrity/           (planned)
```

Notes:

- The Python package lives under `src/insider_gru` (src-layout). Installing the repo (editable install) ensures imports work from `apps/` and `scripts/`.
- `outputs/model` contains the default GRU model artifacts, while `outputs/login/model` contains device/login detector artifacts.

## 6. Installation

### Prerequisites

- Python 3.10+ (recommended: Python 3.12)
- Windows / Linux / macOS

### Create a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The repo uses a src-layout package; install it in editable mode (already included as `-e .` in `requirements.txt`, but safe to run explicitly):

```powershell
pip install -e .
```

## 7. How to Run the Dashboards

This repository contains **5 separate dashboards** that can run independently or simultaneously on different ports.

### Quick Start - Interactive Launcher

```powershell
.\START_DASHBOARDS.ps1
```
This interactive menu lets you choose which dashboard(s) to run, or launch all at once.

---

### Launch 4 Core Dashboards (Single Command)

**Run all 4 core detection dashboards with one command:**

```powershell
.\RUN_4_CORE_DASHBOARDS.ps1
```

This launches in separate windows:
- **NSDM Dashboard** (Port 8505) - CERT r4.2 ensemble detection
- **DEMF Dashboard** (Port 8501) - Ensemble ML anomaly detection
- **HEADS Dashboard** (Port 8503) - Authentication anomaly detection
- **Insider Threat Monitor** (Port 8502) - Real-time threat monitoring

**Access URLs:**
- NSDM: http://localhost:8505
- DEMF: http://localhost:8501
- HEADS: http://localhost:8503
- Insider Threat: http://localhost:8502

---

### Run Dashboards Individually

#### 1. DEMF Anomaly Detection Dashboard (Port 8501)

**Using PowerShell Script:**
```powershell
.\run_demf_dashboard.ps1
```

**Using Direct Command:**
```bash
streamlit run app_streamlit.py --server.port 8501
```

**Features:**
- DEMF (Data-driven Ensemble Machine Learning Framework)
- One-Class SVM + Autoencoder ensemble
- Real-time anomaly scoring
- Feature importance analysis

---

#### 2. Insider Threat Monitoring Dashboard (Port 8502)

**Using PowerShell Script:**
```powershell
.\run_soc_dashboard.ps1
```

**Using Direct Command:**
```bash
streamlit run apps/streamlit_app.py --server.port 8502
```

**Features:**
- AI-driven real-time alerts
- Threat monitoring and verification
- Alert hash integrity (planned)

---

#### 3. HEADS Threat Monitoring Dashboard (Port 8503)

**Using PowerShell Script:**
```powershell
.\run_email_dashboard.ps1
```

**Using Direct Command:**
```bash
streamlit run streamlit_app/app.py --server.port 8503
```

**Features:**
- HEADS (Hybrid Environment Authentication Anomaly Detection)
- Multi-page interface
- Authentication anomaly detection

---

#### 4. Unified HEADS Dashboard (Port 8504)

**Using PowerShell Script:**
```powershell
.\run_unified_dashboard.ps1
```

**Using Direct Command:**
```bash
streamlit run apps/unified_dashboard.py --server.port 8504
```

**Features:**
- All detection modules in one interface
- Cross-module correlation
- Unified alert view
- Comprehensive reporting

---

#### 5. NSDM Insider Threat Dashboard (Port 8505)

**Using PowerShell Script:**
```powershell
.\run_nsdm_dashboard.ps1
```

**Using Direct Command:**
```bash
streamlit run nsdm/dashboard.py --server.port 8505
```

**Features:**
- NSDM (Network Security and Data Mining)
- CERT r4.2 dataset analysis
- Random Forest + Local Outlier Factor ensemble
- Real-time threat monitoring
- Test user analysis (users 701-1000)

---

### Additional Dashboard Options

Email Privilege Misuse Detector (configurable risk scoring + JSON export):

```powershell
streamlit run apps/email_privilege_dashboard.py
```

Login/Device anomaly dashboards:

```powershell
streamlit run apps/login_time_window_dashboard.py
```

SOC / Training dashboards:

```powershell
streamlit run apps/soc_cyber_dashboard.py
streamlit run apps/training_dashboard.py
```

---

### Running Multiple Dashboards Simultaneously

All dashboards run on different ports and can be launched together:

```powershell
# Open 5 terminal windows and run:
streamlit run app_streamlit.py --server.port 8501          # DEMF
streamlit run apps/streamlit_app.py --server.port 8502     # Insider Threat
streamlit run streamlit_app/app.py --server.port 8503      # HEADS
streamlit run apps/unified_dashboard.py --server.port 8504 # Unified
streamlit run nsdm/dashboard.py --server.port 8505         # NSDM
```

Or use the launcher option [6] to open all automatically.

---

### Dashboard Documentation

For detailed dashboard documentation, see:
- **All Dashboards:** [DASHBOARD_README.md](DASHBOARD_README.md)
- **NSDM Specific:** [nsdm/README.md](nsdm/README.md)

## 8. How to Train the GRU Model

### Important: labels

Supervised training requires labels. Your raw CERT-style CSVs may not contain a `label` column by default.
This project supports label attachment via an **id → label mapping CSV** with columns `id,label`.

Generate a starter label mapping template:

```powershell
python scripts/make_label_mapping_template.py
```

Generate labels from user lists:

```powershell
python scripts/build_labels_from_users.py --users-file insider_users.txt --out labels_from_users.csv
```

Generate labels from simple rules (example):

```powershell
python scripts/build_labels_from_rules.py --min-size 3000 --out labels_from_rules.csv
```

Train + evaluate the GRU model using the configured dataset globs and label mapping (see `src/insider_gru/config.py`):

```powershell
python scripts/train_gru.py
```

Artifacts and evaluation outputs are written to:

- `outputs/model/` (e.g., `model.pt`, `preprocessor.joblib`, `meta.json`)
- `outputs/reports/` (e.g., `metrics.json`, `classification_report.txt`, `training_history.*`)
- `outputs/plots/` (e.g., history + confusion matrix)

### Train a device/login GRU detector

This repo includes a dedicated login/device training script with explicit arguments:

```powershell
python scripts/train_login_detector.py `
  --dataset-root Dataset `
  --train-glob "test/R*/device-train-data.csv" `
  --test-glob "train/R*/device-test-data*.csv" `
  --out-dir outputs/login
```

Linux/macOS (Bash):

```bash
python scripts/train_login_detector.py \
  --dataset-root Dataset \
  --train-glob "test/R*/device-train-data.csv" \
  --test-glob "train/R*/device-test-data*.csv" \
  --out-dir outputs/login
```

## 9. How to Train the Transformer Model

Train + evaluate the Transformer model using the same dataset/label configuration as the GRU pipeline:

```powershell
python scripts/train_transformer.py --event-type email
python scripts/train_transformer.py --event-type login
```

Outputs are written under:

- `outputs/transformer/model/` (model weights + metadata)
- `outputs/transformer/reports/` (e.g., `metrics.json`, `classification_report.txt`, `predictions.csv`)

Threshold tuning:

- During training, the script evaluates multiple decision thresholds and selects the best threshold by **validation F1**.
- The per-threshold metrics are saved as `threshold_comparison.csv` in the reports folder.
- The selected threshold is saved into `metrics.json` as `selected_threshold`.

## 10. How to Run Real-Time Simulation

Use the streaming simulator to replay events from a CSV and append live events + alerts to JSONL files:

```powershell
python scripts/simulate_stream.py --event-type email --source Dataset/train/R2/email-train-data.csv --delay 0.25 --clear-output
```

This writes:

- `outputs/live/live_events.jsonl`
- `outputs/live/live_alerts.jsonl`

Then open the monitoring dashboard (it reads those JSONL files):

```powershell
streamlit run apps/streamlit_app.py
```

## 11. How Alert Integrity Verification Works

Planned extension (not yet implemented in the current codebase).

The integrity verification idea is to store a cryptographic hash of each exported alert report (and optionally chain hashes) so that:

- alerts can be audited later
- modifications to exported reports can be detected

Intended storage location:

- `outputs/integrity/` (planned)

## 12. Evaluation Outputs

After training, evaluation artifacts are stored under `outputs/`:

- Model artifacts: `outputs/model/` and/or `outputs/login/model/`
- Plots: `outputs/plots/` and/or `outputs/login/plots/`
- Reports: `outputs/reports/` and/or `outputs/login/reports/`
  - `metrics.json`
  - `classification_report.txt`
  - `training_history.csv` / `training_history.json`

Dashboards also produce JSON reports via download and optional disk persistence (configurable in the sidebar).

## 13. Future Improvements

- Add confidence scoring and calibration (e.g., reliability plots / Platt scaling).
- Improve threshold selection guidance and per-use-case operating points.
- Expand streaming dashboards and add richer analyst drill-down views.
- Expand integrity verification into a dedicated verification dashboard/tool.
- Compare against baselines and ablations to quantify sequential modeling benefits.

# Navigate to project root
cd C:\Users\Ashen\Desktop\Final_year_research

# Step 1: Train the NSDM models (this creates the RF and LOF models)
python nsdm/train_model.py

# Step 2: Generate test user predictions (this creates the dashboard data)
python nsdm/predict_test_users.py

# Step 3: Now run the dashboard
streamlit run nsdm/dashboard.py --server.port 8505