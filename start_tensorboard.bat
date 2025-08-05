@echo off
echo Starting TensorBoard for BraTS 2021 Training...
echo.
echo TensorBoard will open at: http://localhost:6006
echo Press Ctrl+C to stop TensorBoard
echo.
cd /d "X:\Projects\brats_medical_segmentation"
tensorboard --logdir=logs_brats2021 --port=6006 --host=localhost
pause