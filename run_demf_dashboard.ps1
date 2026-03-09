# DEMF Dashboard Runner
# This runs the DEMF anomaly detection dashboard from sahan-dev branch

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting DEMF Anomaly Detection Dashboard" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

# Run the DEMF streamlit dashboard
streamlit run app_streamlit.py --server.port 8501
