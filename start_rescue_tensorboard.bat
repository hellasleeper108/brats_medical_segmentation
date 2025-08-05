@echo off
echo ================================================
echo    TENSORBOARD FOR RESCUE TRAINING
echo ================================================
echo.
echo Stopping any existing TensorBoard...
taskkill /f /im "tensorboard.exe" 2>nul
timeout /t 2 /nobreak >nul
echo.
echo Starting TensorBoard for RESCUE training logs...
echo.
echo TensorBoard will open at: http://localhost:6007
echo (Using port 6007 to avoid conflicts)
echo.
echo Press Ctrl+C to stop TensorBoard
echo.
cd /d "X:\Projects\brats_medical_segmentation"
tensorboard --logdir=logs_brats2021_rescue --port=6007 --host=localhost