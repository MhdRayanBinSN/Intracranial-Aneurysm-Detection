@echo off
REM MedGemma Demo Runner
REM This script sets up the environment and runs the demo

echo ============================================
echo     MedGemma Demo Runner
echo ============================================

REM Set HuggingFace Token
set HF_TOKEN=hf_vgPPpqQGhENjEcrEsalFsxvONXYWHEyVE1

REM Activate conda environment
call C:\Users\Rayan\anaconda3\Scripts\activate.bat medgemma

REM Run demo
python demo.py

pause
