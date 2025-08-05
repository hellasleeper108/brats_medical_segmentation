@echo off
echo STARTING BRATS 2021 RESCUE TRAINING
echo ===================================
echo.
echo This version has safety improvements:
echo - Lower learning rate (0.0001)
echo - Stricter gradient clipping
echo - Linear LR decay (no OneCycleLR)
echo - Frequent checkpoints
echo.
echo Press any key to start rescue training...
pause
echo.
cd /d "X:\Projects\brats_medical_segmentation"
python train_brats2021_rescue.py
pause