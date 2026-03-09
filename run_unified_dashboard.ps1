# Unified Dashboard Runner
# This runs the unified HEADS dashboard with all detection modules

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Unified HEADS Dashboard" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: http://localhost:8504" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

# Run the unified dashboard
streamlit run apps/unified_dashboard.py --server.port 8504
