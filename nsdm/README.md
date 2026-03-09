# NSDM - Network Security and Data Mining
## Insider Threat Monitoring System

This is the NSDM (Network Security and Data Mining) insider threat detection system from the malith_dev branch, integrated into the merge_branch as a separate, standalone dashboard.

---

## 📊 Overview

NSDM uses ensemble machine learning (Random Forest + Local Outlier Factor) to detect insider threats in the CERT r4.2 dataset. It provides real-time monitoring and alert generation for unseen test users.

---

## 🚀 Quick Start

### Run the NSDM Dashboard:

**Using PowerShell Script:**
```powershell
.\run_nsdm_dashboard.ps1
```

**Using Direct Streamlit Command:**
```bash
streamlit run nsdm/dashboard.py --server.port 8505
```

**Access:** http://localhost:8505

---

## 📁 File Structure

```
nsdm/
├── config.py              # Configuration and paths
├── dashboard.py           # Main Streamlit dashboard
├── train_model.py         # Model training script
├── feature_extractor.py   # Feature engineering
├── predict_test_users.py  # Prediction on test users
├── evaluate_test_model.py # Model evaluation
├── realtime_monitor.py    # Real-time monitoring
└── privacy_utils.py       # Privacy utilities

models/
├── cert_r42_rf_model.pkl         # Random Forest model
├── cert_r42_lof_model.pkl        # LOF anomaly detector
├── cert_r42_scaler.pkl           # Feature scaler
├── realtime_alerts.json          # Live alerts
├── realtime_state.json           # Monitoring state
└── train_metrics.txt             # Training metrics

data/r4.2/
├── license.txt
└── readme.txt
```

---

## 🔧 Features

### Dashboard Capabilities:
- **Live Alert Monitoring** - Real-time threat alerts with auto-refresh
- **Test User Dashboard** - Comprehensive view of unseen test users
- **Threat Scoring** - Ensemble scoring (RF + LOF)
- **Alert Prioritization** - Critical, High, Medium, Low severity levels
- **Interactive Visualizations** - Plotly-powered threat analysis
- **Privacy-Preserving** - Anonymized user IDs

### Detection Models:
- **Random Forest Classifier** - Supervised learning for known patterns
- **Local Outlier Factor (LOF)** - Unsupervised anomaly detection
- **Ensemble Scoring** - Combined threat scoring

---

## 📊 Thresholds

- **Critical Threat:** ≥ 0.90
- **High Threat:** ≥ 0.70
- **Medium Threat:** < 0.70

Configurable via environment variables:
```bash
export NSADM_THREAT_THRESHOLD=0.70
export NSADM_CRITICAL_THRESHOLD=0.90
```

---

## 🔬 Usage Workflow

### 1. Train Models
```bash
cd nsdm
python train_model.py
```

### 2. Extract Features
```bash
python feature_extractor.py
```

### 3. Make Predictions
```bash
python predict_test_users.py
```

### 4. Run Dashboard
```powershell
.\run_nsdm_dashboard.ps1
```

### 5. (Optional) Real-time Monitor
```bash
python realtime_monitor.py
```

---

## 🎯 Dashboard Views

### 1. Live Alerts Tab
- Real-time threat monitoring
- Auto-refresh capability
- Alert details with severity levels
- Timestamp tracking

### 2. Test Users Dashboard
- 300 unseen test users (user_701 - user_1000)
- Threat score distribution
- Top threats identification
- User behavior analysis

---

## 📋 Data Requirements

**CERT r4.2 Dataset:**
- Place CERT r4.2 data files in `data/r4.2/`
- Dataset contains 1000 synthetic users
- Training: users 1-700
- Testing: users 701-1000

---

## 🔒 Privacy Features

Uses `privacy_utils.py` for:
- User ID anonymization
- Sensitive data protection
- Secure logging

---

## ⚙️ Configuration

Edit `nsdm/config.py` to adjust:
- Data paths
- Model paths
- User split (train/test)
- Thresholds
- Monitor intervals

### Environment Variables:
```bash
CERT_DATA_DIR         # Data directory
NSADM_MODELS_DIR      # Models directory
NSADM_OUTPUTS_DIR     # Outputs directory
NSADM_THREAT_THRESHOLD    # Threat threshold (default: 0.70)
NSADM_CRITICAL_THRESHOLD  # Critical threshold (default: 0.90)
NSADM_MONITOR_INTERVAL    # Monitor interval in seconds (default: 3)
```

---

## 🐛 Troubleshooting

### Dashboard doesn't load:
```bash
# Check if models exist
ls models/cert_r42_*.pkl

# If missing, train models first
cd nsdm
python train_model.py
```

### No alerts showing:
```bash
# Run predictions first
cd nsdm
python predict_test_users.py

# Or start realtime monitor
python realtime_monitor.py
```

### Port already in use:
```bash
# Use different port
streamlit run nsdm/dashboard.py --server.port 8506
```

---

## 📈 Model Performance

Check `models/train_metrics.txt` for:
- Accuracy scores
- Precision/Recall
- F1 scores
- ROC-AUC metrics

---

## 🆚 Comparison with Other Dashboards

| Dashboard | Focus | Port | Models |
|-----------|-------|------|--------|
| **NSDM** | CERT r4.2 Ensemble Detection | 8505 | RF + LOF |
| DEMF | General Anomaly Detection | 8501 | OCSVM + Autoencoder |
| SOC | Real-time Threat Monitoring | 8502 | Various |
| Email | Email Privilege Misuse | 8503 | GRU |
| Unified | All-in-one | 8504 | Combined |

---

## 📝 Notes

- **NSDM is completely separate** from other dashboards
- Uses its own models in `models/cert_r42_*.pkl`
- Runs independently on port 8505
- No code mixing with DEMF, SOC, Email, or Unified dashboards
- Specifically designed for CERT r4.2 dataset analysis

---

## 🔗 Related Files

- Main README: `DASHBOARD_README.md`
- DEMF Dashboard: `app_streamlit.py`
- Other dashboards: `apps/` directory
- Unified launcher: `START_DASHBOARDS.ps1`
