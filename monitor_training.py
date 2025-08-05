#!/usr/bin/env python3
"""
Real-time training monitoring script for BraTS 2021 model.
"""

import os
import time
import json
import psutil
import GPUtil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class TrainingMonitor:
    """Monitor training progress, GPU usage, and performance metrics."""
    
    def __init__(self, log_dir='logs_brats2021', checkpoint_dir='checkpoints_brats2021'):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.start_time = datetime.now()
        self.monitoring_data = {
            'timestamps': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'ram_usage': [],
            'train_loss': [],
            'val_loss': [],
            'dice_score': [],
            'learning_rate': []
        }
        
        # Create monitoring directory
        self.monitor_dir = Path('monitoring')
        self.monitor_dir.mkdir(exist_ok=True)
        
    def get_system_stats(self):
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
    
    def parse_training_logs(self):
        """Parse training logs for loss and metrics."""
        log_file = self.log_dir / 'training.log'
        if not log_file.exists():
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            latest_metrics = {}
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if 'Epoch' in line and 'Loss' in line:
                    # Parse epoch info
                    if 'Train Loss:' in line:
                        try:
                            loss = float(line.split('Train Loss:')[1].split(',')[0].strip())
                            latest_metrics['train_loss'] = loss
                        except:
                            pass
                    
                    if 'Val Loss:' in line:
                        try:
                            loss = float(line.split('Val Loss:')[1].split(',')[0].strip())
                            latest_metrics['val_loss'] = loss
                        except:
                            pass
                    
                    if 'Dice:' in line:
                        try:
                            dice = float(line.split('Dice:')[1].split(',')[0].strip())
                            latest_metrics['dice_score'] = dice
                        except:
                            pass
                
                if len(latest_metrics) >= 3:  # Found all metrics
                    break
            
            return latest_metrics
        except Exception as e:
            print(f"Error parsing logs: {e}")
            return None
    
    def check_checkpoints(self):
        """Check for saved checkpoints."""
        if not self.checkpoint_dir.exists():
            return []
        
        checkpoints = list(self.checkpoint_dir.glob('*.pth'))
        checkpoint_info = []
        
        for cp in checkpoints:
            stat = cp.stat()
            checkpoint_info.append({
                'name': cp.name,
                'size': stat.st_size / (1024**2),  # MB
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'path': str(cp)
            })
        
        return sorted(checkpoint_info, key=lambda x: x['modified'], reverse=True)
    
    def estimate_completion_time(self, current_epoch, total_epochs=100):
        """Estimate training completion time."""
        if current_epoch <= 0:
            return "Calculating..."
        
        elapsed = datetime.now() - self.start_time
        time_per_epoch = elapsed / current_epoch
        remaining_epochs = total_epochs - current_epoch
        eta = datetime.now() + (time_per_epoch * remaining_epochs)
        
        return eta.strftime("%Y-%m-%d %H:%M:%S")
    
    def create_monitoring_dashboard(self):
        """Create a monitoring dashboard."""
        current_time = datetime.now()
        stats = self.get_system_stats()
        training_metrics = self.parse_training_logs()
        checkpoints = self.check_checkpoints()
        
        # Store data for plotting
        self.monitoring_data['timestamps'].append(current_time)
        if stats:
            self.monitoring_data['gpu_usage'].append(stats['gpu_usage'])
            self.monitoring_data['gpu_memory'].append(stats['gpu_memory'])
            self.monitoring_data['cpu_usage'].append(stats['cpu_usage'])
            self.monitoring_data['ram_usage'].append(stats['ram_usage'])
        
        if training_metrics:
            self.monitoring_data['train_loss'].append(training_metrics.get('train_loss', np.nan))
            self.monitoring_data['val_loss'].append(training_metrics.get('val_loss', np.nan))
            self.monitoring_data['dice_score'].append(training_metrics.get('dice_score', np.nan))
        
        # Generate report
        report = {
            'timestamp': current_time.isoformat(),
            'elapsed_time': str(current_time - self.start_time),
            'system_stats': stats,
            'training_metrics': training_metrics,
            'checkpoints': checkpoints[:5],  # Latest 5 checkpoints
            'monitoring_urls': {
                'tensorboard': 'http://localhost:6006',
                'training_script': 'train_brats2021.py'
            }
        }
        
        return report
    
    def save_monitoring_plot(self):
        """Save monitoring plots."""
        if len(self.monitoring_data['timestamps']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('BraTS 2021 Training Monitoring', fontsize=16)
        
        timestamps = self.monitoring_data['timestamps']
        
        # GPU Usage
        axes[0, 0].plot(timestamps, self.monitoring_data['gpu_usage'], 'b-', label='GPU Usage %')
        axes[0, 0].plot(timestamps, self.monitoring_data['gpu_memory'], 'r-', label='GPU Memory %')
        axes[0, 0].set_title('GPU Utilization')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # System Resources
        axes[0, 1].plot(timestamps, self.monitoring_data['cpu_usage'], 'g-', label='CPU %')
        axes[0, 1].plot(timestamps, self.monitoring_data['ram_usage'], 'm-', label='RAM %')
        axes[0, 1].set_title('System Resources')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training Loss
        if any(not np.isnan(x) for x in self.monitoring_data['train_loss']):
            valid_indices = [i for i, x in enumerate(self.monitoring_data['train_loss']) if not np.isnan(x)]
            if valid_indices:
                valid_times = [timestamps[i] for i in valid_indices]
                valid_train = [self.monitoring_data['train_loss'][i] for i in valid_indices]
                valid_val = [self.monitoring_data['val_loss'][i] for i in valid_indices if not np.isnan(self.monitoring_data['val_loss'][i])]
                
                axes[1, 0].plot(valid_times, valid_train, 'b-', label='Train Loss')
                if len(valid_val) == len(valid_times):
                    axes[1, 0].plot(valid_times, valid_val, 'r-', label='Val Loss')
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
        
        # Dice Score
        if any(not np.isnan(x) for x in self.monitoring_data['dice_score']):
            valid_indices = [i for i, x in enumerate(self.monitoring_data['dice_score']) if not np.isnan(x)]
            if valid_indices:
                valid_times = [timestamps[i] for i in valid_indices]
                valid_dice = [self.monitoring_data['dice_score'][i] for i in valid_indices]
                
                axes[1, 1].plot(valid_times, valid_dice, 'g-', label='Dice Score')
                axes[1, 1].axhline(y=0.85, color='r', linestyle='--', label='Target: 0.85')
                axes[1, 1].set_title('Dice Score')
                axes[1, 1].set_ylabel('Dice Coefficient')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        # Format x-axis
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.monitor_dir / 'training_monitor.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_status_report(self, report):
        """Print a formatted status report."""
        print("\n" + "="*80)
        print("🧠 BraTS 2021 TRAINING MONITOR")
        print("="*80)
        print(f"📅 Time: {report['timestamp'][:19]}")
        print(f"⏱️  Elapsed: {report['elapsed_time']}")
        
        if report['system_stats']:
            stats = report['system_stats']
            print(f"\n🖥️  SYSTEM STATUS:")
            print(f"   GPU Usage: {stats['gpu_usage']:.1f}% | Memory: {stats['gpu_memory']:.1f}%")
            print(f"   CPU Usage: {stats['cpu_usage']:.1f}% | RAM: {stats['ram_usage']:.1f}%")
            if 'gpu_temp' in stats:
                print(f"   GPU Temp: {stats['gpu_temp']:.1f}°C")
        
        if report['training_metrics']:
            metrics = report['training_metrics']
            print(f"\n📊 TRAINING METRICS:")
            if 'train_loss' in metrics:
                print(f"   Train Loss: {metrics['train_loss']:.4f}")
            if 'val_loss' in metrics:
                print(f"   Val Loss: {metrics['val_loss']:.4f}")
            if 'dice_score' in metrics:
                dice = metrics['dice_score']
                status = "🎯 EXCELLENT" if dice > 0.85 else "📈 IMPROVING" if dice > 0.7 else "🔄 LEARNING"
                print(f"   Dice Score: {dice:.4f} {status}")
        
        if report['checkpoints']:
            print(f"\n💾 CHECKPOINTS ({len(report['checkpoints'])}):")
            for cp in report['checkpoints'][:3]:
                print(f"   {cp['name']} - {cp['size']:.1f}MB - {cp['modified'].strftime('%H:%M:%S')}")
        
        print(f"\n🔗 MONITORING LINKS:")
        print(f"   TensorBoard: {report['monitoring_urls']['tensorboard']}")
        print(f"   Training Plot: monitoring/training_monitor.png")
        print("="*80)


def main():
    """Main monitoring loop."""
    monitor = TrainingMonitor()
    
    print("🚀 Starting BraTS 2021 Training Monitoring...")
    print("📊 TensorBoard: http://localhost:6006")
    print("⏱️  Monitoring every 30 seconds...")
    print("🛑 Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            report = monitor.create_monitoring_dashboard()
            monitor.print_status_report(report)
            monitor.save_monitoring_plot()
            
            # Save report to file
            with open(monitor.monitor_dir / 'latest_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            time.sleep(30)  # Monitor every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print("📊 Final monitoring plot saved to: monitoring/training_monitor.png")


if __name__ == "__main__":
    main()