# Quick Start - Run All Dashboards
# This script helps you choose which dashboard to run

Write-Host ""
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   HEADS - Insider Threat Detection System     ║" -ForegroundColor Cyan
Write-Host "║           Dashboard Launcher                   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Available Dashboards:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [1] DEMF Anomaly Detection Dashboard" -ForegroundColor Green
Write-Host "      → Port 8501" -ForegroundColor Gray
Write-Host "      → Ensemble ML anomaly detection" -ForegroundColor Gray
Write-Host ""
Write-Host "  [2] SOC/Cyber Threat Dashboard" -ForegroundColor Green
Write-Host "      → Port 8502" -ForegroundColor Gray
Write-Host "      → Real-time threat monitoring" -ForegroundColor Gray
Write-Host ""
Write-Host "  [3] Email Privilege Misuse Dashboard" -ForegroundColor Green
Write-Host "      → Port 8503" -ForegroundColor Gray
Write-Host "      → Email behavior analysis" -ForegroundColor Gray
Write-Host ""
Write-Host "  [4] Unified HEADS Dashboard" -ForegroundColor Green
Write-Host "      → Port 8504" -ForegroundColor Gray
Write-Host "      → All detection modules combined" -ForegroundColor Gray
Write-Host ""
Write-Host "  [5] NSDM Insider Threat Dashboard" -ForegroundColor Green
Write-Host "      → Port 8505" -ForegroundColor Gray
Write-Host "      → CERT r4.2 ensemble detection (RF + LOF)" -ForegroundColor Gray
Write-Host ""
Write-Host "  [6] Run All Dashboards" -ForegroundColor Magenta
Write-Host "      → Opens all dashboards simultaneously" -ForegroundColor Gray
Write-Host ""
Write-Host "  [Q] Quit" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Select a dashboard (1-6 or Q)"

switch ($choice.ToUpper()) {
    "1" {
        Write-Host ""
        Write-Host "Launching DEMF Dashboard..." -ForegroundColor Green
        .\run_demf_dashboard.ps1
    }
    "2" {
        Write-Host ""
        Write-Host "Launching SOC Dashboard..." -ForegroundColor Green
        .\run_soc_dashboard.ps1
    }
    "3" {
        Write-Host ""
        Write-Host "Launching Email Dashboard..." -ForegroundColor Green
        .\run_email_dashboard.ps1
    }
    "4" {
        Write-Host ""
        Write-Host "Launching Unified Dashboard..." -ForegroundColor Green
        .\run_unified_dashboard.ps1
    }
    "5" {
        Write-Host ""
        Write-Host "Launching NSDM Dashboard..." -ForegroundColor Green
        .\run_nsdm_dashboard.ps1
    }
    "6" {
        Write-Host ""
        Write-Host "Launching ALL Dashboards..." -ForegroundColor Magenta
        Write-Host ""
        Write-Host "Opening dashboards in separate windows..." -ForegroundColor Yellow
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\run_demf_dashboard.ps1"
        Start-Sleep -Seconds 2
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\run_soc_dashboard.ps1"
        Start-Sleep -Seconds 2
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\run_email_dashboard.ps1"
        Start-Sleep -Seconds 2
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\run_unified_dashboard.ps1"
        Start-Sleep -Seconds 2
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\run_nsdm_dashboard.ps1"
        
        Write-Host ""
        Write-Host "✓ All dashboards started!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Access URLs:" -ForegroundColor Cyan
        Write-Host "  • DEMF:       http://localhost:8501" -ForegroundColor White
        Write-Host "  • SOC:        http://localhost:8502" -ForegroundColor White
        Write-Host "  • Email:      http://localhost:8503" -ForegroundColor White
        Write-Host "  • Unified:    http://localhost:8504" -ForegroundColor White
        Write-Host "  • NSDM:       http://localhost:8505" -ForegroundColor White
        Write-Host ""
        Write-Host "Press any key to exit launcher..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    "Q" {
        Write-Host ""
        Write-Host "Goodbye!" -ForegroundColor Yellow
        exit
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
        Start-Sleep -Seconds 2
    }
}
