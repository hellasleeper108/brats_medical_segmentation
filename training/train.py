import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import wandb
from tensorboardX import SummaryWriter

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet3D, VisionTransformer3D
from data import BraTSDataset, COVIDCTDataset, get_transforms
from utils.metrics import SegmentationMetrics, DiceScore
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.device import setup_device
from training.losses import DiceLoss, CombinedLoss


class Trainer:
    """Medical image segmentation trainer."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = setup_device(config['hardware']['device'])
        self.setup_logging()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_metrics()
        
        # Mixed precision training
        self.scaler = GradScaler() if config['training']['mixed_precision'] else None
        
        # Logging
        self.writer = None
        if config['logging']['use_tensorboard']:
            log_dir = os.path.join(config['paths']['logs'], 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        if config['logging']['use_wandb']:
            wandb.init(project='medical-segmentation', config=config)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.config['paths']['logs']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self):
        """Setup model architecture."""
        model_config = self.config['model']
        
        if model_config['name'] == 'unet':
            self.model = UNet3D(
                in_channels=1,
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout']
            )
        elif model_config['name'] == 'vit':
            self.model = VisionTransformer3D(
                img_size=tuple(model_config['input_size']),
                in_channels=1,
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout']
            )
        else:
            raise ValueError(f"Unsupported model: {model_config['name']}")
        
        self.model = self.model.to(self.device)
        
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model_config['name']}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model size: {self.model.get_model_size():.2f} MB")
    
    def setup_data(self):
        """Setup data loaders."""
        data_config = self.config['data']
        
        # Get transforms
        train_transforms = get_transforms('train', 
                                        roi_size=tuple(self.config['model']['input_size']))
        val_transforms = get_transforms('val',
                                      roi_size=tuple(self.config['model']['input_size']))
        
        # Create datasets
        if data_config['dataset'] == 'brats':
            train_dataset = BraTSDataset(
                data_dir=os.path.join(data_config['data_root'], 'train'),
                transforms=train_transforms,
                cache_rate=data_config['cache_rate']
            )
            val_dataset = BraTSDataset(
                data_dir=os.path.join(data_config['data_root'], 'val'),
                transforms=val_transforms,
                cache_rate=data_config['cache_rate']
            )
        elif data_config['dataset'] == 'covid_ct':
            train_dataset = COVIDCTDataset(
                data_dir=os.path.join(data_config['data_root'], 'train'),
                transforms=train_transforms,
                cache_rate=data_config['cache_rate']
            )
            val_dataset = COVIDCTDataset(
                data_dir=os.path.join(data_config['data_root'], 'val'),
                transforms=val_transforms,
                cache_rate=data_config['cache_rate']
            )
        else:
            raise ValueError(f"Unsupported dataset: {data_config['dataset']}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            persistent_workers=self.config['hardware']['persistent_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Use batch size 1 for validation
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
    
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        training_config = self.config['training']
        
        if training_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
        
        # Setup scheduler
        if training_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs']
            )
        elif training_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['epochs'] // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Setup loss function
        self.criterion = CombinedLoss(
            num_classes=self.config['model']['num_classes'],
            include_background=False
        )
    
    def setup_metrics(self):
        """Setup evaluation metrics."""
        self.metrics = SegmentationMetrics(
            num_classes=self.config['model']['num_classes'],
            include_background=False
        )
        self.dice_metric = DiceScore(include_background=False)
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.config['training']['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.config['training']['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = self.dice_metric(outputs, labels)
                total_dice += dice.item()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
            
            # Log to tensorboard
            if self.writer:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
                self.writer.add_scalar('Train/Dice_Batch', dice.item(), global_step)
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc='Validation'):
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                dice = self.dice_metric(outputs, labels)
                comprehensive_metrics = self.metrics.calculate_all_metrics(
                    torch.argmax(outputs, dim=1), labels.squeeze(1)
                )
                
                total_loss += loss.item()
                total_dice += dice.item()
                all_metrics.append(comprehensive_metrics)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        # Average comprehensive metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'per_class':
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'sensitivity': avg_metrics.get('sensitivity', 0),
            'specificity': avg_metrics.get('specificity', 0)
        }
    
    def train(self):
        """Main training loop."""
        best_dice = 0
        patience = 0
        max_patience = 20
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Dice: {train_metrics['dice']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Dice: {val_metrics['dice']:.4f}"
            )
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('Train/Loss_Epoch', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Dice_Epoch', train_metrics['dice'], epoch)
                self.writer.add_scalar('Val/Loss_Epoch', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Dice_Epoch', val_metrics['dice'], epoch)
                self.writer.add_scalar('Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
            
            # Wandb logging
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_dice': train_metrics['dice'],
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
                patience = 0
                
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_dice': best_dice,
                    'config': self.config
                }, is_best, self.config['paths']['checkpoints'])
                
                self.logger.info(f"New best Dice score: {best_dice:.4f}")
            else:
                patience += 1
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_dice': best_dice,
                    'config': self.config
                }, False, self.config['paths']['checkpoints'], f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if patience >= max_patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        self.logger.info(f"Training completed. Best Dice score: {best_dice:.4f}")
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(args.resume, trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed from checkpoint: {args.resume}")
        print(f"Best Dice score: {checkpoint['best_dice']:.4f}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()