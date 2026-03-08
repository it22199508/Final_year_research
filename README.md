# DEMF Simplified (One-Folder Project)

This project is a **simplified, easy-to-track** prototype of your DEMF idea.

✅ Only 3 main files:
- `DEMF_ML_Pipeline.ipynb` (full ML pipeline in one notebook)
- `demf_core.py` (shared helper code)
- `app_streamlit.py` (Streamlit monitoring dashboard)

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Put your dataset files

Copy your CERT-style logs into:

```
data/raw/
  logon.csv
  device.csv
  http.csv
  LDAP/   (optional)
```

Your columns can be mixed case / have extra spaces — the loader normalizes them.

## 3) Run the notebook

Open `DEMF_ML_Pipeline.ipynb` and run cells top-to-bottom.

## 4) Run the Streamlit dashboard

```bash
streamlit run app_streamlit.py
```

Then click **Train + Detect** (or **Detect Only** if you already trained).

## Sample data

A small sample dataset is included in `data/sample_raw/` so you can test immediately.

To use it:
- In `config.yaml`, set `data.raw_dir: "data/sample_raw"`
- Or in the Streamlit sidebar, set Raw data folder to `data/sample_raw`


## Dashboard features

The Streamlit dashboard includes:
- Overview KPIs + alert timeline
- Alert Explorer with filters + CSV download
- User Investigation (score trend, day drilldown, feature deviation, hourly heatmap, top domains)
- Behavior Analytics (risk leaderboard, scatter plots, domains on alert-days)
- Model diagnostics (reconstruction vs SVM anomaly + artifact inventory)

## Optional: F1 evaluation (needs labels)

Create a labels CSV with:
- `user_hash` (or `user`)
- `day` (or `date` / `timestamp`)
- `label` (0/1)

Then set `evaluation.labels_path` in `config.yaml` or paste the path into the Streamlit sidebar field **Labels CSV (optional)**.
Open the **Evaluation (F1)** tab to see F1/precision/recall and the confusion matrix.
