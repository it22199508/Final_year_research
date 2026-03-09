# HEADS — Hybrid Environment Authentication Anomaly Detection System (Transformer + GraphSAGE + XGBoost)

This project implements the architecture shown in your diagram:

**Raw IAM / security logs → preprocessing → SMOTE balancing →**
1) **Transformer Autoencoder** (temporal anomaly score)  
2) **GraphSAGE GNN** (relational anomaly score)  
**→ XGBoost classifier → trained anomaly detection model**

> ✅ This repo is ready to run against the dataset you uploaded as `cybersecurity.csv`.
> Even though the CSV columns look like network/web telemetry, the pipeline treats each `src_ip` as an “actor”
> and models authentication/access behaviour using temporal + relational signals.

---

## 1) What it detects (mapped to your objectives)

- **Account takeover / credential abuse**: unusual request patterns, ports, URLs, rapid behavioural drift (temporal AE score ↑)
- **Privilege abuse / unauthorized access**: actor connects to unusual destinations/resources compared to its neighbourhood (GNN score ↑)
- **“Impossible travel” analogue**: rapid context shifts for an actor (e.g., sudden dst_ip/url diversity spikes in short windows)

---

## 2) Repo structure

```
HEADS_IAM_Anomaly_Project/
  data/
    raw/                 # put the CSV here
    processed/
  models/
    transformer_ae.pt
    graphsage.pt
    xgb.json
  notebooks/
    01_eda_and_features.ipynb
    02_train_transformer_autoencoder.ipynb
    03_train_gnn_graphsage.ipynb
    04_train_xgboost.ipynb
  src/
    heads/
      config.py
      data.py
      features.py
      smote.py
      temporal_ae.py
      graph.py
      gnn.py
      xgb.py
      train_all.py
      serve_api.py
  requirements.txt
```

---

## 3) Quick start

### 3.1 Put your dataset in place

Copy your CSV to:

```
data/raw/cybersecurity.csv
```

> In this sandbox, the original file is at: `/mnt/data/cybersecurity.csv`

### 3.2 Install dependencies

Create a venv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Torch Geometric note**
- Torch Geometric wheels depend on your Torch/CUDA version.
- Follow the official install selector if pip fails.

### 3.3 Train everything (end-to-end)

```bash
python -m src.heads.train_all --data data/raw/cybersecurity.csv
```

This produces:
- `models/transformer_ae.pt`
- `models/graphsage.pt`
- `models/xgb.json`

### 3.4 Real-time scoring API (FastAPI)

```bash
python -m src.heads.serve_api --host 0.0.0.0 --port 8000
```

POST an event JSON to `/score`.

---

## 4) Outputs (what you’ll get)

For each event, the system returns:
- `temporal_score` (Transformer AE reconstruction error)
- `relational_score` (GraphSAGE neighbourhood deviation score)
- `xgb_proba` (probability of malicious)
- `prediction` (0 benign / 1 malicious)

---

## 5) Dataset assumptions

The pipeline expects at least these columns:

- `timestamp` (string/datetime)
- `src_ip`, `dst_ip` (strings)
- `src_port`, `dst_port` (ints)
- `protocol` (categorical)
- `bytes_sent`, `bytes_received` (numeric)
- `user_agent` (string)
- `url` (string; may be missing)
- `is_internal_traffic` (bool)
- `label` (0/1, optional but recommended)
- `attack_type` (string, optional)

If `label` is missing, the code can still compute anomaly scores, but XGBoost will not be trained supervised.

---

## 6) License

MIT (for coursework / learning use).


## Streamlit dashboard

Run:

```bash
streamlit run streamlit_app/app.py
```


## Single training notebook

Open and run:
- `notebooks/00_HEADS_single_pipeline.ipynb`


## Training (recommended)

Open and run: `HEADS_Train_and_Save_Models.ipynb`


## Train/Test + Model Variants
Run `HEADS_Train_Test_Variants.ipynb` to generate:
- Train/test splits in `data/test/`
- Multiple XGBoost variants in `models/variants/`
- Scored test instances `data/test/test_*__xgb_v*.csv`

## Dummy Live Feed (simulation)
Run:
`python scripts/generate_live_feed.py --out data/live/live_events.csv --rows 300 --interval 5`
Then open dashboard, choose **Live feed**, enable Auto refresh.
