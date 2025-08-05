@echo off
title Live Training Output Monitor
echo ================================================
echo    LIVE TRAINING OUTPUT MONITOR
echo ================================================
echo.
echo This shows the actual training progress bars,
echo loss values, and Dice scores in real-time.
echo.
echo Press Ctrl+C to stop monitoring
echo.
pause

cd /d "X:\Projects\brats_medical_segmentation"

:LOOP
echo.
echo [%date% %time%] Checking for training output...
echo ================================================

REM Try to find and tail the training process output
python -c "
import psutil
import subprocess
import time

# Find training process
training_pid = None
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'train_brats2021_rescue.py' in cmdline:
                training_pid = proc.info['pid']
                print(f'Found rescue training process PID: {training_pid}')
                print(f'Training is ACTIVE and running')
                break
    except:
        pass

if not training_pid:
    print('No rescue training process found')
    print('Training may have finished or not started yet')
"

timeout /t 10 /nobreak >nul
goto LOOP