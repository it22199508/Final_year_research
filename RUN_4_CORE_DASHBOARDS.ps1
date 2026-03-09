# Run 4 Core Dashboards
# Launches NSDM, DEMF, HEADS, and Insider Threat Monitoring dashboards

Write-Host ""
Write-Host "Starting 4 Core Detection Dashboards..." -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/4] Launching NSDM Dashboard on port 8505..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; streamlit run nsdm/dashboard.py --server.port 8505"
Start-Sleep -Seconds 2

Write-Host "[2/4] Launching DEMF Dashboard on port 8501..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; streamlit run app_streamlit.py --server.port 8501"
Start-Sleep -Seconds 2

Write-Host "[3/4] Launching HEADS Dashboard on port 8503..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; streamlit run streamlit_app/app.py --server.port 8503"
Start-Sleep -Seconds 2

Write-Host "[4/4] Launching Insider Threat Monitoring on port 8502..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; streamlit run apps/streamlit_app.py --server.port 8502"

Write-Host ""
Write-Host "All 4 dashboards started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "  NSDM:              http://localhost:8505" -ForegroundColor White
Write-Host "  DEMF:              http://localhost:8501" -ForegroundColor White
Write-Host "  HEADS:             http://localhost:8503" -ForegroundColor White
Write-Host "  Insider Threat:    http://localhost:8502" -ForegroundColor White
Write-Host ""
