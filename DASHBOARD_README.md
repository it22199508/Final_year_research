# Dashboard Guide

This repository contains **FIVE separate dashboards** that can run independently or simultaneously:

---

## 🚀 Quick Start

### Easy Way - Use the Launcher:
```powershell
.\START_DASHBOARDS.ps1
```
This interactive menu lets you choose which dashboard(s) to run.

### Manual Way - Run Individual Dashboards:

#### Using PowerShell Scripts (Recommended):

**1. DEMF Anomaly Detection Dashboard**
```powershell
.\run_demf_dashboard.ps1
```
**Port:** 8501 | **URL:** http://localhost:8501

**2. SOC/Cyber Threat Dashboard**
```powershell
.\run_soc_dashboard.ps1
```
**Port:** 8502 | **URL:** http://localhost:8502

**3. Email Privilege Misuse Dashboard**
```powershell
.\run_email_dashboard.ps1
```
**Port:** 8503 | **URL:** http://localhost:8503

**4. Unified HEADS Dashboard**
```powershell
.\run_unified_dashboard.ps1
```
**Port:** 8504 | **URL:** http://localhost:8504

**5. NSDM Insider Threat Dashboard**
```powershell
.\run_nsdm_dashboard.ps1
```
**Port:** 8505 | **URL:** http://localhost:8505

---

#### Using Direct Streamlit Commands:

If PowerShell scripts are disabled, use these direct commands:

**1. DEMF Anomaly Detection Dashboard**
```bash
streamlit run app_streamlit.py
```
Opens the DEMF ensemble ML anomaly detection dashboard
- **Default Port:** 8501
- **Features:** One-Class SVM, Autoencoder, real-time scoring

**2. Insider Threat Monitoring Dashboard**
```bash
streamlit run apps/streamlit_app.py
```
Opens the AI-Driven Insider Threat Monitoring dashboard
- **Default Port:** 8501 (use `--server.port 8502` to change)
- **Features:** Real-time alerts, threat monitoring, alert verification

**3. HEADS Threat Monitoring Dashboard**
```bash
streamlit run streamlit_app/app.py
```
Opens the HEADS (Hybrid Environment Authentication Anomaly Detection) dashboard
- **Default Port:** 8501 (use `--server.port 8503` to change)
- **Features:** Authentication anomaly detection, multi-page interface

**4. NSDM Insider Threat Dashboard**
```bash
streamlit run nsdm/dashboard.py
```
Opens the NSDM (Network Security and Data Mining) dashboard
- **Default Port:** 8501 (use `--server.port 8505` to change)
- **Features:** CERT r4.2 ensemble detection, RF + LOF models

**To run on specific ports:**
```bash
streamlit run app_streamlit.py --server.port 8501
streamlit run apps/streamlit_app.py --server.port 8502
streamlit run streamlit_app/app.py --server.port 8503
streamlit run nsdm/dashboard.py --server.port 8505
```

---

## 📊 Dashboard Details

### 1. DEMF Dashboard (from sahan-dev branch)

**Purpose**: DEMF (Data-driven Ensemble Machine Learning Framework) anomaly detection

**Features**:
- Ensemble ML anomaly detection
- One-Class SVM and Autoencoder models
- Real-time scoring and alerts
- Feature importance analysis

**Main Files**:
- `app_streamlit.py` - Dashboard application
- `demf_core.py` - Core DEMF detection logic
- `config.yaml` - Configuration file
- `DEMF_ML_Pipeline.ipynb` - Training pipeline

**Models**: 
- `models/current/autoencoder.joblib`
- `models/current/ocsvm.joblib`
- `models/current/scaler.joblib`

---

### 2. SOC/Cyber Threat Dashboard

**Purpose**: Security Operations Center real-time threat monitoring

**Features**:
- Real-time threat detection
- Alert prioritization
- Timeline visualization
- Auto-refresh monitoring

**Main Files**:
- `apps/soc_cyber_dashboard.py`

**Data**: Uses processed threat data from `reports/current/`

---

### 3. Email Privilege Misuse Dashboard

**Purpose**: Email behavior analysis and privilege misuse detection

**Features**:
- GRU-based email behavior model
- Privilege escalation detection
- Suspicious email pattern analysis
- User risk scoring

**Main Files**:
- `apps/email_privilege_dashboard.py`
- `src/insider_gru/` - GRU model implementation

**Models**: GRU classifier for email behavior

---

### 4. Unified HEADS Dashboard

**Purpose**: Comprehensive view combining all detection modules

**Features**:
- All detection modules in one interface
- Cross-module correlation
- Unified alert view
- Comprehensive reporting

**Main Files**:
- `apps/unified_dashboard.py`

---

### 5. NSDM Dashboard (from malith_dev branch)

**Purpose**: Network Security and Data Mining - CERT r4.2 Insider Threat Detection

**Features**:
- Ensemble ML detection (Random Forest + LOF)
- Real-time alert monitoring
- Test user analysis (users 701-1000)
- Threat score prioritization
- Privacy-preserving analytics

**Main Files**:
- `nsdm/dashboard.py` - Main dashboard
- `nsdm/config.py` - Configuration
- `nsdm/train_model.py` - Model training
- `nsdm/realtime_monitor.py` - Real-time monitoring
- `nsdm/README.md` - Detailed documentation

**Models**:
- `models/cert_r42_rf_model.pkl` - Random Forest
- `models/cert_r42_lof_model.pkl` - Local Outlier Factor
- `models/cert_r42_scaler.pkl` - Feature scaler

**Data**: CERT r4.2 dataset (700 training users, 300 test users)

---

## 🔧 Requirements

Install dependencies:
```powershell
pip install -r requirements.txt
```

Essential packages:
- `streamlit` - Dashboard framework
- `pandas`, `numpy` - Data processing
- `scikit-learn` - ML models
- `torch` - Deep learning (for GRU)
- `plotly` - Visualizations

---

## 📁 Data Requirements

Make sure you have data in these locations:

**For DEMF Dashboard:**
- `data/raw/logon.csv` - Login data
- `data/raw/device.csv` - Device data
- `data/raw/http.csv` - HTTP data
- `data/raw/LDAP/` - LDAP data
- `models/current/` - Trained models

**For Other Dashboards:**
- `reports/current/scored_events.csv` - Scored events
- `data/processed/` - Processed data files

**Sample Data:**
- `data/sample_raw/` - Small sample dataset for testing

---

## 🎯 Running All Dashboards Simultaneously

Each dashboard runs on a different port, so you can run all at once:

```powershell
# Open 4 PowerShell terminals and run:
# Terminal 1:
.\run_demf_dashboard.ps1

# Terminal 2:
.\run_soc_dashboard.ps1

# Terminal 3:
.\run_email_dashboard.ps1

# Terminal 4:
.\run_unified_dashboard.ps1
```

Or use the launcher option 5 to open all automatically:
```powershell
.\START_DASHBOARDS.ps1
# Select option [5]
```

**Access All:**
- DEMF: http://localhost:8501
- SOC: http://localhost:8502  
- Email: http://localhost:8503
- Unified: http://localhost:8504

---

## ⚠️ Important Notes

**Do NOT mix code between branches:**
- `sahan-dev` code: `app_streamlit.py`, `demf_core.py`, etc.
- `merge_branch` code: `apps/` directory files

**Large files are gitignored:**
- CSV files in `data/`, `reports/`
- Model files (`.joblib`)
- These stay local or use Git LFS

**Memory Usage:**
- Running all dashboards simultaneously requires ~2-4GB RAM
- Each dashboard loads its own models and data
- Close unused dashboards to free memory

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Find and kill process on port (e.g., 8501)
netstat -ano | findstr "8501"
taskkill /PID <process_id> /F
```

### Missing Models
```powershell
# Train models using notebooks
jupyter notebook DEMF_ML_Pipeline.ipynb
```

### Import Errors
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Dashboard Won't Load
1. Check if data files exist in `data/raw/`
2. Verify models exist in `models/current/`
3. Check terminal for error messages
4. Try sample data in `data/sample_raw/`

---

## 📖 Training Models

### DEMF Models:
```powershell
# Open Jupyter and run:
jupyter notebook DEMF_ML_Pipeline.ipynb
```

### GRU Email Models:
See `apps/training_dashboard.py` or relevant notebooks in `notebooks/`

---

## 🔍 Which Dashboard Should I Use?

- **Testing/Demo** → Use sample data + DEMF Dashboard
- **Anomaly Detection** → DEMF Dashboard  
- **Real-time SOC Monitoring** → SOC Dashboard
- **Email Analysis** → Email Privilege Dashboard
- **Everything Combined** → Unified Dashboard
- **Development/Debug** → Run individual dashboards
