# SOC/Cyber Threat Dashboard Runner
# This runs the SOC insider threat detection dashboard

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting SOC/Cyber Threat Dashboard" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: http://localhost:8502" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

# Run the SOC dashboard
streamlit run apps/soc_cyber_dashboard.py --server.port 8502
