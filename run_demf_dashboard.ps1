# DEMF Dashboard Runner
# This runs the DEMF anomaly detection dashboard from sahan-dev branch

Write-Host "Starting DEMF Dashboard..." -ForegroundColor Green
Write-Host "Dashboard will be available at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

# Run the DEMF streamlit dashboard
streamlit run app_streamlit.py --server.port 8501
