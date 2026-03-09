# Email Privilege Dashboard Runner
# This runs the email privilege misuse detection dashboard

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Email Privilege Misuse Dashboard" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: http://localhost:8503" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

# Run the email privilege dashboard
streamlit run apps/email_privilege_dashboard.py --server.port 8503
