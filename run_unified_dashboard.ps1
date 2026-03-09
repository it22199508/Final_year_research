# Unified Dashboard Runner
# This runs the unified HEADS dashboard with all detection modules

Write-Host "Starting Unified HEADS Dashboard..." -ForegroundColor Green
Write-Host "Dashboard will be available at http://localhost:8502" -ForegroundColor Cyan
Write-Host ""

# Run the unified dashboard on a different port to avoid conflicts
streamlit run apps/unified_dashboard.py --server.port 8502
