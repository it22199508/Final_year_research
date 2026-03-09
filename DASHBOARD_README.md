# Dashboard Guide

This repository contains **two separate dashboards** that can run independently:

## 1. DEMF Dashboard (from sahan-dev branch)

**Purpose**: DEMF (Data-driven Ensemble Machine Learning Framework) anomaly detection dashboard

**Files**:
- `app_streamlit.py` - Main dashboard application
- `demf_core.py` - Core DEMF detection logic
- `config.yaml` - Configuration file
- `make_labels.py` - Label generation utility
- `DEMF_ML_Pipeline.ipynb` - Training pipeline notebook

**To Run**:
```powershell
.\run_demf_dashboard.ps1
```
Or manually:
```powershell
streamlit run app_streamlit.py --server.port 8501
```

**Access**: http://localhost:8501

---

## 2. Unified HEADS Dashboard (from merge_branch)

**Purpose**: Unified Hybrid Ensemble Anomaly Detection System with multiple threat detection modules

**Files**:
- `apps/unified_dashboard.py` - Main unified dashboard
- `apps/soc_cyber_dashboard.py` - SOC/Cyber threat view
- `apps/email_privilege_dashboard.py` - Email privilege monitoring
- `apps/after_hours_login_dashboard.py` - After-hours detection
- `apps/training_dashboard.py` - Model training interface

**To Run**:
```powershell
.\run_unified_dashboard.ps1
```
Or manually:
```powershell
streamlit run apps/unified_dashboard.py --server.port 8502
```

**Access**: http://localhost:8502

---

## Running Both Dashboards Simultaneously

You can run both dashboards at the same time since they use different ports:
1. Open terminal 1: `.\run_demf_dashboard.ps1` (port 8501)
2. Open terminal 2: `.\run_unified_dashboard.ps1` (port 8502)

---

## Requirements

Install dependencies:
```powershell
pip install -r requirements.txt
```

Make sure you have the necessary data files in:
- `data/raw/` - Raw cybersecurity data
- `models/current/` - Trained models (or train new ones using notebooks)
- `reports/current/` - Will be generated when running evaluations

---

## Important Notes

⚠️ **Do NOT mix code between branches**:
- `sahan-dev` code: `app_streamlit.py`, `demf_core.py`, etc.
- `merge_branch` code: `apps/` directory files

⚠️ **Large files are gitignored**:
- CSV files in `data/`, `reports/`
- Model files (`.joblib`)
- Store these locally or use Git LFS if needed

---

## Troubleshooting

**Port already in use**:
```powershell
# Find process using port 8501 or 8502
netstat -ano | findstr "8501"
# Kill the process
taskkill /PID <process_id> /F
```

**Missing dependencies**:
```powershell
pip install --upgrade streamlit pandas numpy scikit-learn
```
