@echo off
title BraTS 2021 Rescue Training Monitor
echo ========================================
echo    BraTS 2021 RESCUE TRAINING MONITOR
echo ========================================
echo.
echo This will show real-time training status:
echo - GPU usage and temperature
echo - Training process status  
echo - Latest Dice scores
echo - Checkpoint files
echo.
echo Press Ctrl+C to stop monitoring
echo.
pause
cd /d "X:\Projects\brats_medical_segmentation"
python simple_monitor.py