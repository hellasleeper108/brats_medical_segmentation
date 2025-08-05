#!/usr/bin/env python3
"""
Simple training monitoring script for BraTS 2021 model.
"""

import os
import time
import json
import psutil
import GPUtil
from pathlib import Path
from datetime import datetime

def get_system_stats():
    """Get current system resource usage."""
    try:
        # GPU stats
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # RTX 4080 Super
            gpu_usage = gpu.load * 100
            gpu_memory = gpu.memoryUtil * 100
            gpu_temp = gpu.temperature
        else:
            gpu_usage = gpu_memory = gpu_temp = 0
        
        # CPU and RAM
        cpu_usage = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        
        return {
            'gpu_usage': gpu_usage,
            'gpu_memory': gpu_memory,
            'gpu_temp': gpu_temp,
            'cpu_usage': cpu_usage,
            'ram_usage': ram_usage,
            'available_ram': ram.available / (1024**3)  # GB
        }
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return None

def check_training_status():
    """Check if training is still running."""
    try:
        # Check for python processes running train_brats2021.py
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'train_brats2021.py' in cmdline:
                        return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False, None
    except Exception as e:
        print(f"Error checking training status: {e}")
        return False, None

def check_checkpoints():
    """Check for saved checkpoints."""
    checkpoint_dir = Path('checkpoints_brats2021')
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    checkpoint_info = []
    
    for cp in checkpoints:
        stat = cp.stat()
        checkpoint_info.append({
            'name': cp.name,
            'size_mb': stat.st_size / (1024**2),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S'),
            'path': str(cp)
        })
    
    return sorted(checkpoint_info, key=lambda x: x['modified'], reverse=True)

def parse_latest_log():
    """Parse the latest training log."""
    log_file = Path('logs_brats2021/training.log')
    if not log_file.exists():
        return "No log file found"
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Get last few lines for current status
        recent_lines = lines[-10:] if len(lines) > 10 else lines
        return '\n'.join([line.strip() for line in recent_lines if line.strip()])
    except Exception as e:
        return f"Error reading log: {e}"

def print_status_report():
    """Print a comprehensive status report."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*80)
    print("BraTS 2021 TRAINING MONITOR")
    print("="*80)
    print(f"Time: {current_time}")
    
    # Training Status
    is_running, pid = check_training_status()
    if is_running:
        print(f"Status: TRAINING ACTIVE (PID: {pid})")
    else:
        print("Status: TRAINING NOT DETECTED")
    
    # System Stats
    stats = get_system_stats()
    if stats:
        print(f"\nSYSTEM RESOURCES:")
        print(f"  GPU Usage: {stats['gpu_usage']:.1f}%")
        print(f"  GPU Memory: {stats['gpu_memory']:.1f}%") 
        print(f"  GPU Temp: {stats['gpu_temp']:.1f}C")
        print(f"  CPU Usage: {stats['cpu_usage']:.1f}%")
        print(f"  RAM Usage: {stats['ram_usage']:.1f}%")
        print(f"  Available RAM: {stats['available_ram']:.1f} GB")
    
    # Checkpoints
    checkpoints = check_checkpoints()
    if checkpoints:
        print(f"\nCHECKPOINTS ({len(checkpoints)}):")
        for cp in checkpoints[:3]:  # Show latest 3
            print(f"  {cp['name']} - {cp['size_mb']:.1f}MB - {cp['modified']}")
    else:
        print("\nCHECKPOINTS: None found yet")
    
    # Recent logs
    recent_log = parse_latest_log()
    print(f"\nRECENT LOG OUTPUT:")
    print("-" * 40)
    print(recent_log[-500:])  # Last 500 characters
    print("-" * 40)
    
    print(f"\nMONITORING LINKS:")
    print(f"  TensorBoard: http://localhost:6006")
    print(f"  Log file: logs_brats2021/training.log")
    
    print("="*80)

def main():
    """Main monitoring loop."""
    print("Starting BraTS 2021 Training Monitoring...")
    print("TensorBoard: http://localhost:6006")
    print("Monitoring every 30 seconds...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print_status_report()
            time.sleep(30)  # Monitor every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

if __name__ == "__main__":
    main()