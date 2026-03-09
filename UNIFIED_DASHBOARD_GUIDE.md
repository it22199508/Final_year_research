# 🛡️ Unified Threat Monitoring Dashboard - User Guide

## Overview

The unified dashboard combines two powerful threat detection systems into a single interface:

1. **AI-Driven Insider Threat Monitoring Dashboard** - GRU-based sequential anomaly detection for email and login activity
2. **Threat Monitoring Live (HEADS)** - Hybrid Environment Authentication Anomaly Detection System using Transformer + GraphSAGE + XGBoost

## Quick Start

### Access the Dashboard

The unified dashboard is now running at: **http://localhost:8501**

### Navigation

Use the sidebar navigation buttons to switch between the two systems:

- **📊 AI-Driven Insider Threat Monitoring Dashboard** - Monitor insider threats from email/login sequences
- **🔴 Threat Monitoring Live (HEADS)** - Real-time network behavior analysis with live feed

## System 1: AI-Driven Insider Threat Monitoring

### Features
- Real-time event and alert monitoring
- JSONL-based event streaming
- Risk scoring and categorization
- Alert integrity verification

### How to Use

1. **Start the Event Stream Simulator:**
   ```powershell
   # For email events:
   python scripts/simulate_stream.py --event-type email --source Dataset/train/R2/email-train-data.csv --delay 0.25 --clear-output
   
   # For login events:
   python scripts/simulate_stream.py --event-type login --source Dataset/train/R2/device-train-data.csv --delay 0.25 --clear-output
   ```

2. **View Live Events:** The dashboard will automatically display events from:
   - `outputs/live/live_events.jsonl`
   - `outputs/live/live_alerts.jsonl`

3. **Monitor Alerts:** High-risk alerts are highlighted in red/orange based on risk score

## System 2: Threat Monitoring Live (HEADS)

### Features
- Real-time threat categorization
- Live feed visualization
- Threat distribution charts
- Configurable threat mapping profiles
- Auto-refresh capability

### How to Use

1. **Live Feed (Already Running):**
   The live feed generator is already running in the background, creating simulated threat data every 5 seconds.

2. **Configuration Options (in sidebar):**
   - **Data Source:** Choose between Live feed, Test instances, or Default CSV
   - **Threat Mapping Profile:** Select threat detection sensitivity
   - **Min Probability:** Filter events by probability threshold
   - **Rows to Show:** Control table size
   - **Auto Refresh:** Enable/disable automatic data refresh
   - **Refresh Interval:** Set refresh rate (2-30 seconds)

3. **Threat Categories:**
   - 🔴 **Account Takeover** - High probability events
   - 🟠 **Privilege Abuse** - Relational anomalies
   - 🟡 **Suspicious Login** - Temporal anomalies
   - 🟢 **Normal** - Benign activity

## Currently Running Services

✅ **Unified Dashboard:** http://localhost:8501  
✅ **Live Feed Generator:** Running in background (updates every 5 seconds)  
✅ **Data Directory:** `data/live/live_events.csv` (301 lines generated)

## Manual Controls

### Start/Stop Live Feed Generator

**Start:**
```powershell
python scripts/generate_live_feed.py --out data/live/live_events.csv --rows 300 --interval 5
```

**Stop:**
```powershell
# Press Ctrl+C in the terminal running the generator
```

### Start/Stop Unified Dashboard

**Start:**
```powershell
streamlit run apps/unified_dashboard.py --server.port 8501
```

**Stop:**
```powershell
# Press Ctrl+C in the terminal running the dashboard
# Or close all streamlit processes:
Get-Process | Where-Object {$_.Path -like "*streamlit*"} | Stop-Process -Force
```

## Tips for Best Experience

1. **Enable Auto-Refresh** in HEADS system to see live threat updates
2. **Adjust Refresh Interval** based on your needs (5 seconds recommended)
3. **Use Dark Mode** for better visibility
4. **Filter by Probability** to focus on high-risk events
5. **Switch Between Views** using sidebar navigation buttons

## Troubleshooting

### Dashboard not loading?
- Check if port 8501 is available
- Restart the dashboard: `streamlit run apps/unified_dashboard.py`

### No live data showing?
- Verify live feed generator is running
- Check `data/live/live_events.csv` exists and is being updated
- Manually refresh using the 🔄 button

### Events not updating in Insider Threat view?
- Start the simulate_stream.py script
- Ensure `outputs/live/` directory exists
- Check JSONL files are being written

## Architecture

```
Unified Dashboard
├── Navigation Sidebar
│   ├── AI-Driven Insider Threat Monitoring
│   │   ├── Event Stream (from simulate_stream.py)
│   │   ├── Alert Detection (GRU-based)
│   │   └── Risk Scoring
│   └── Threat Monitoring Live (HEADS)
│       ├── Live Feed (from generate_live_feed.py)
│       ├── Threat Categorization
│       └── Visualization
└── Real-time Updates
```

## Next Steps

1. ✅ Explore both dashboards using the navigation buttons
2. ✅ Start the Insider Threat event simulator for full functionality
3. ✅ Adjust threat detection parameters in the sidebar
4. ✅ Monitor live threats in real-time with auto-refresh enabled

Enjoy your unified threat monitoring experience! 🛡️
