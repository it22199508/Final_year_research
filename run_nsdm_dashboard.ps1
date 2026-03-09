# NSDM Dashboard Runner
# This runs the NSDM (Network Security and Data Mining) Insider Threat Monitoring System

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting NSDM Insider Threat Dashboard" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: http://localhost:8505" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

# Run the NSDM dashboard
streamlit run nsdm/dashboard.py --server.port 8505
