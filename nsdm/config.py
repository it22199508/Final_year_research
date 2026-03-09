# nsdm/config.py

import os
from pathlib import Path

# Base paths - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.getenv("CERT_DATA_DIR", str(PROJECT_ROOT / "data" / "r4.2"))
MODELS_DIR = os.getenv("NSADM_MODELS_DIR", str(PROJECT_ROOT / "models"))
OUTPUTS_DIR = os.getenv("NSADM_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))

# User split for CERT r4.2
TRAIN_USER_OFFSET = 0
TRAIN_USER_COUNT = 700

TEST_USER_OFFSET = 700
TEST_USER_COUNT = 300

# Feature files
TRAIN_FEATURES_FILE = os.path.join(MODELS_DIR, "cert_r42_features_train.csv")
TEST_FEATURES_FILE = os.path.join(MODELS_DIR, "cert_r42_features_test.csv")

# Model files
RF_MODEL_FILE = os.path.join(MODELS_DIR, "cert_r42_rf_model.pkl")
LOF_MODEL_FILE = os.path.join(MODELS_DIR, "cert_r42_lof_model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "cert_r42_scaler.pkl")

# Prediction files
TRAIN_PRED_FILE = os.path.join(MODELS_DIR, "cert_r42_train_predictions.csv")
TEST_PRED_FILE = os.path.join(MODELS_DIR, "cert_r42_test_predictions.csv")
TEST_DASHBOARD_FILE = os.path.join(MODELS_DIR, "cert_r42_dashboard_test.csv")

# Realtime files
REALTIME_ALERTS_FILE = os.path.join(MODELS_DIR, "realtime_alerts.json")
REALTIME_STATE_FILE = os.path.join(MODELS_DIR, "realtime_state.json")

# Thresholds
THREAT_THRESHOLD = float(os.getenv("NSADM_THREAT_THRESHOLD", "0.70"))
CRITICAL_THRESHOLD = float(os.getenv("NSADM_CRITICAL_THRESHOLD", "0.90"))

# Realtime settings
MONITOR_INTERVAL = float(os.getenv("NSADM_MONITOR_INTERVAL", "3"))
TOP_N_LOOP = int(os.getenv("NSADM_TOP_N_LOOP", "5"))