# MedGemma Demo Runner (PowerShell)
# Run this script with: .\run_demo.ps1

Write-Host "============================================"
Write-Host "     MedGemma Demo Runner (PowerShell)"
Write-Host "============================================"

# Set HuggingFace Token
$env:HF_TOKEN = "hf_vgPPpqQGhENjEcrEsalFsxvONXYWHEyVE1"
Write-Host "HF_TOKEN set: $($env:HF_TOKEN.Substring(0, 10))..."

# Activate conda and run
conda activate medgemma
python demo.py
