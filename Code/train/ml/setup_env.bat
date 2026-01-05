@echo off
REM =============================================================================
REM Intracranial Aneurysm Detection - Environment Setup Script
REM For Windows with CUDA 11.8 and RTX 3060
REM =============================================================================

echo ============================================================
echo  Setting up Aneurysm Detection Environment
echo  CUDA Version: 11.8
echo  GPU: RTX 3060 (6GB VRAM)
echo ============================================================
echo.

REM Create new conda environment
echo [1/4] Creating conda environment 'aneurysm' with Python 3.10...
call conda create -n aneurysm python=3.10 -y

REM Activate environment
echo.
echo [2/4] Activating environment...
call conda activate aneurysm

REM Install PyTorch with CUDA 11.8
echo.
echo [3/4] Installing PyTorch with CUDA 11.8 support...
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

REM Install other dependencies
echo.
echo [4/4] Installing remaining dependencies...
pip install pydicom nibabel
pip install numpy pandas scipy
pip install scikit-learn scikit-image
pip install pyyaml matplotlib tensorboard tqdm
pip install albumentations opencv-python-headless pillow

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo To use this environment, run:
echo   conda activate aneurysm
echo.
echo To start training, run:
echo   cd ml
echo   python train.py
echo.
pause
