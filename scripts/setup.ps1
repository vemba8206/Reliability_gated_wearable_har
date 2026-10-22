Write-Host "Setting up Reliability-Gated HAR Environment..." -ForegroundColor Cyan
python -m pip install -r requirements.txt
python -m src.data_download --synthetic
Write-Host "Setup complete!" -ForegroundColor Green
