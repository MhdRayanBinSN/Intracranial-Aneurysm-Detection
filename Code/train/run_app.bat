@echo off
title Intracranial Aneurysm Detection App
color 0B

echo ===================================================
echo   Intracranial Aneurysm Detection App
echo   Backend: FastAPI + MedGemma + BiomedCLIP
echo   Frontend: React + Vite
echo ===================================================
echo.

:: ---- Configuration ----
set DATA_ROOT=C:\Users\Rayan\Desktop\Main Project
set DATASET_ZIP=%DATA_ROOT%\rsna-intracranial-aneurysm-detection.zip
set BACKEND_PORT=8000
set FRONTEND_PORT=5173

:: ---- Check conda ----
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] conda not found in PATH. Make sure Anaconda/Miniconda is installed.
    pause
    exit /b 1
)

:: ---- Activate conda environment ----
echo [1/4] Activating conda environment...
call conda activate medgemma 2>nul
if %errorlevel% neq 0 (
    echo [!] 'medgemma' env not found. Trying 'aneurysm'...
    call conda activate aneurysm 2>nul
    if %errorlevel% neq 0 (
        echo [!] No conda env found. Using base environment.
        call conda activate base
    )
)
echo       Active env: %CONDA_DEFAULT_ENV%
echo.

:: ---- Check Python deps ----
echo [2/4] Checking Python dependencies...
python -c "import fastapi; import torch; print('       FastAPI + PyTorch OK')" 2>nul
if %errorlevel% neq 0 (
    echo [!] Missing Python dependencies. Installing...
    pip install -r backend\requirements.txt
    pip install -r medgemma\requirements.txt
)
echo.

:: ---- Start Backend ----
echo [3/4] Starting Backend on port %BACKEND_PORT%...
echo       MedGemma model loading takes 1-2 minutes on first run.
echo.
start "Backend - FastAPI" cmd /k "cd /d %~dp0backend && set DATA_ROOT=%DATA_ROOT% && set DATASET_ZIP=%DATASET_ZIP% && python -u main.py"

:: Wait for backend to be ready
echo       Waiting for backend to start...
timeout /t 5 /nobreak >nul

:: ---- Start Frontend ----
echo [4/4] Starting Frontend on port %FRONTEND_PORT%...
echo.
start "Frontend - React" cmd /k "cd /d %~dp0frontend && npm run dev"

:: Wait a moment for Vite to start
timeout /t 3 /nobreak >nul

echo.
echo ===================================================
echo   App is starting!
echo.
echo   Frontend:  http://localhost:%FRONTEND_PORT%
echo   Backend:   http://localhost:%BACKEND_PORT%
echo   API Docs:  http://localhost:%BACKEND_PORT%/docs
echo.
echo   NOTE: MedGemma model takes 1-2 min to load.
echo         The /health endpoint will show model_loaded=true
echo         once it's ready.
echo ===================================================
echo.
echo Press any key to open the app in your browser...
pause >nul
start http://localhost:%FRONTEND_PORT%
