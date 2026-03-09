# Run 4 Core Dashboards
# Launches NSDM, DEMF, HEADS, and Insider Threat Monitoring dashboards in separate windows

Write-Host ""
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Launching 4 Core Detection Dashboards     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting dashboards in separate windows..." -ForegroundColor Yellow
Write-Host ""

# Dashboard 1: NSDM (Port 8505)
Write-Host "[1/4] Launching NSDM Dashboard (Port 8505)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'NSDM Dashboard - Port 8505' -ForegroundColor Cyan; streamlit run nsdm/dashboard.py --server.port 8505"
Start-Sleep -Seconds 2

# Dashboard 2: DEMF (Port 8501)
Write-Host "[2/4] Launching DEMF Dashboard (Port 8501)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'DEMF Dashboard - Port 8501' -ForegroundColor Cyan; streamlit run app_streamlit.py --server.port 8501"
Start-Sleep -Seconds 2

# Dashboard 3: HEADS Threat Monitoring (Port 8503)
Write-Host "[3/4] Launching HEADS Dashboard (Port 8503)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'HEADS Dashboard - Port 8503' -ForegroundColor Cyan; streamlit run streamlit_app/app.py --server.port 8503"
Start-Sleep -Seconds 2

# Dashboard 4: Insider Threat Monitoring (Port 8502)
Write-Host "[4/4] Launching Insider Threat Monitoring (Port 8502)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'Insider Threat Monitoring - Port 8502' -ForegroundColor Cyan; streamlit run apps/streamlit_app.py --server.port 8502"

Write-Host ""
Write-Host "✓ All 4 dashboards started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "  • NSDM:                    http://localhost:8505" -ForegroundColor White
Write-Host "  • DEMF:                    http://localhost:8501" -ForegroundColor White
Write-Host "  • HEADS:                   http://localhost:8503" -ForegroundColor White
Write-Host "  • Insider Threat Monitor:  http://localhost:8502" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit launcher..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
