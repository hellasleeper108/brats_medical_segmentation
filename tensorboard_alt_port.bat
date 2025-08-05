@echo off
echo Starting TensorBoard on alternative port 6007...
echo.
echo TensorBoard will open at: http://localhost:6007
echo Press Ctrl+C to stop TensorBoard
echo.
cd /d "X:\Projects\brats_medical_segmentation"
tensorboard --logdir=logs_brats2021 --port=6007 --host=localhost
pause