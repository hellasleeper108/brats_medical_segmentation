import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import numpy as np
import os
import time
from typing import Dict, Tuple, Optional, List
import yaml

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet3D, VisionTransformer3D
from utils.checkpoint import load_checkpoint
from utils.device import get_device


class QuantizedUNet3D(nn.Module):
    """Quantization-ready U-Net 3D model."""
    
    def __init__(self, base_model: UNet3D):
        super(QuantizedUNet3D, self).__init__()
        self.base_model = base_model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.base_model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse conv-bn-relu patterns for better quantization."""
        for module_name, module in self.base_model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for conv-bn-relu patterns
                layers = list(module.children())
                if len(layers) >= 2:
                    for i in range(len(layers) - 1):
                        if (isinstance(layers[i], nn.Conv3d) and 
                            isinstance(layers[i + 1], nn.BatchNorm3d)):
                            if i + 2 < len(layers) and isinstance(layers[i + 2], nn.ReLU):
                                # Conv-BN-ReLU pattern
                                fuse_modules(module, [str(i), str(i + 1), str(i + 2)], inplace=True)
                            else:
                                # Conv-BN pattern
                                fuse_modules(module, [str(i), str(i + 1)], inplace=True)


class ModelQuantizer:
    """Medical image segmentation model quantizer."""
    
    def __init__(self, model_path: str, model_type: str = 'unet', 
                 config_path: str = 'configs/config.yaml'):
        """
        Initialize the quantizer.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('unet', 'vit')
            config_path: Path to model configuration
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = get_device('cpu')  # Quantization works on CPU
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load original model
        self.original_model = self._load_original_model()
        self.original_size = self._get_model_size(self.original_model)
        
        print(f"Original model size: {self.original_size:.2f} MB")
    
    def _load_original_model(self):
        """Load the original trained model."""
        model_config = self.config['model']
        
        if self.model_type == 'unet':
            model = UNet3D(
                in_channels=1,
                num_classes=model_config['num_classes'],
                dropout=0.0  # Disable dropout for inference
            )
        elif self.model_type == 'vit':
            model = VisionTransformer3D(
                img_size=tuple(model_config['input_size']),
                in_channels=1,
                num_classes=model_config['num_classes'],
                dropout=0.0  # Disable dropout for inference
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load weights
        checkpoint = load_checkpoint(self.model_path, self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model.to(self.device)
    
    def _get_model_size(self, model):
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def quantize_dynamic(self, dtype=torch.qint8) -> torch.nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Args:
            dtype: Quantization data type
        
        Returns:
            Dynamically quantized model
        """
        print("Applying dynamic quantization...")
        
        # Dynamic quantization
        if self.model_type == 'unet':
            quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Conv3d, nn.Linear},  # Specify layers to quantize
                dtype=dtype
            )
        else:  # ViT
            quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear},  # Focus on linear layers for transformers
                dtype=dtype
            )
        
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = self.original_size / quantized_size
        
        print(f"Dynamic quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_model
    
    def quantize_static(self, calibration_data: List[torch.Tensor]) -> torch.nn.Module:
        """
        Apply static quantization to the model.
        
        Args:
            calibration_data: List of calibration tensors
        
        Returns:
            Statically quantized model
        """
        print("Applying static quantization...")
        
        if self.model_type != 'unet':
            print("Warning: Static quantization is primarily tested with U-Net")
        
        # Prepare model for quantization
        quantized_model = QuantizedUNet3D(self.original_model)
        quantized_model.eval()
        
        # Fuse layers
        quantized_model.fuse_model()
        
        # Set quantization configuration
        quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        torch.quantization.prepare(quantized_model, inplace=True)
        
        # Calibration
        print("Running calibration...")
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                if i >= 10:  # Limit calibration samples
                    break
                _ = quantized_model(data)
                if i % 5 == 0:
                    print(f"  Calibration sample {i+1}")
        
        # Convert to quantized model
        torch.quantization.convert(quantized_model, inplace=True)
        
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = self.original_size / quantized_size
        
        print(f"Static quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_model
    
    def prune_model(self, pruning_ratio: float = 0.3) -> torch.nn.Module:
        """
        Apply magnitude-based pruning to the model.
        
        Args:
            pruning_ratio: Fraction of weights to prune
        
        Returns:
            Pruned model
        """
        print(f"Applying magnitude-based pruning ({pruning_ratio*100:.1f}%)...")
        
        import torch.nn.utils.prune as prune
        
        # Create a copy of the model
        pruned_model = type(self.original_model)(
            **{k: v for k, v in self.original_model.__dict__.items() 
               if not k.startswith('_')}
        )
        pruned_model.load_state_dict(self.original_model.state_dict())
        
        # Apply pruning to Conv3d and Linear layers
        modules_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                modules_to_prune.append((module, 'weight'))
        
        # Global magnitude-based pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )
        
        # Remove pruning reparameterization
        for module, _ in modules_to_prune:
            prune.remove(module, 'weight')
        
        # Calculate sparsity
        total_params = sum(p.numel() for p in pruned_model.parameters())
        zero_params = sum((p == 0).sum().item() for p in pruned_model.parameters())
        sparsity = zero_params / total_params
        
        print(f"Model sparsity: {sparsity*100:.2f}%")
        
        return pruned_model
    
    def benchmark_models(self, models: Dict[str, torch.nn.Module], 
                        test_input: torch.Tensor, 
                        num_runs: int = 50) -> Dict[str, Dict]:
        """
        Benchmark different model variants.
        
        Args:
            models: Dictionary of model variants
            test_input: Test input tensor
            num_runs: Number of benchmark runs
        
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        print(f"Benchmarking models with {num_runs} runs...")
        
        for name, model in models.items():
            print(f"  Benchmarking {name}...")
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    output = model(test_input)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            model_size = self._get_model_size(model)
            
            results[name] = {
                'avg_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'model_size_mb': model_size,
                'throughput_fps': 1.0 / np.mean(times),
                'output_shape': list(output.shape)
            }
            
            print(f"    Avg time: {results[name]['avg_time']*1000:.2f} ms")
            print(f"    Model size: {model_size:.2f} MB")
            print(f"    Throughput: {results[name]['throughput_fps']:.2f} FPS")
        
        return results
    
    def evaluate_accuracy(self, models: Dict[str, torch.nn.Module],
                         test_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Dict]:
        """
        Evaluate accuracy of different model variants.
        
        Args:
            models: Dictionary of model variants
            test_data: List of (input, target) tuples
        
        Returns:
            Dictionary with accuracy results
        """
        from utils.metrics import DiceScore
        
        results = {}
        dice_metric = DiceScore(include_background=False)
        
        print("Evaluating model accuracy...")
        
        for name, model in models.items():
            print(f"  Evaluating {name}...")
            
            dice_scores = []
            
            with torch.no_grad():
                for i, (input_tensor, target_tensor) in enumerate(test_data):
                    if i >= 20:  # Limit evaluation samples
                        break
                    
                    output = model(input_tensor)
                    dice = dice_metric(output, target_tensor)
                    dice_scores.append(dice.item())
            
            dice_scores = np.array(dice_scores)
            
            results[name] = {
                'avg_dice': float(np.mean(dice_scores)),
                'std_dice': float(np.std(dice_scores)),
                'min_dice': float(np.min(dice_scores)),
                'max_dice': float(np.max(dice_scores)),
                'num_samples': len(dice_scores)
            }
            
            print(f"    Avg Dice: {results[name]['avg_dice']:.4f}")
        
        return results
    
    def save_optimized_model(self, model: torch.nn.Module, 
                           save_path: str, 
                           optimization_type: str) -> None:
        """
        Save optimized model.
        
        Args:
            model: Optimized model
            save_path: Path to save the model
            optimization_type: Type of optimization applied
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        if optimization_type == 'dynamic_quantization':
            torch.jit.save(torch.jit.script(model), save_path)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimization_type': optimization_type,
                'original_model_size': self.original_size,
                'optimized_model_size': self._get_model_size(model),
                'config': self.config
            }, save_path)
        
        print(f"Saved {optimization_type} model to {save_path}")


def create_calibration_data(data_loader, num_samples: int = 20) -> List[torch.Tensor]:
    """
    Create calibration data for static quantization.
    
    Args:
        data_loader: DataLoader with calibration data
        num_samples: Number of calibration samples
    
    Returns:
        List of calibration tensors
    """
    calibration_data = []
    
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
        
        if isinstance(batch, dict):
            # MONAI format
            input_tensor = batch['image']
        else:
            # Standard format
            input_tensor = batch[0]
        
        calibration_data.append(input_tensor)
    
    return calibration_data


def main():
    """Main function for model optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize medical segmentation model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='unet',
                       choices=['unet', 'vit'],
                       help='Type of model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='optimized_models',
                       help='Directory to save optimized models')
    parser.add_argument('--optimization', type=str, 
                       choices=['dynamic', 'static', 'pruning', 'all'],
                       default='all',
                       help='Type of optimization to apply')
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = ModelQuantizer(args.model_path, args.model_type, args.config)
    
    # Create test input
    input_size = quantizer.config['model']['input_size']
    test_input = torch.randn(1, 1, *input_size)
    
    models = {'original': quantizer.original_model}
    
    # Apply optimizations
    if args.optimization in ['dynamic', 'all']:
        dynamic_model = quantizer.quantize_dynamic()
        models['dynamic_quantized'] = dynamic_model
        
        save_path = os.path.join(args.output_dir, f'{args.model_type}_dynamic_quantized.pth')
        quantizer.save_optimized_model(dynamic_model, save_path, 'dynamic_quantization')
    
    if args.optimization in ['pruning', 'all']:
        pruned_model = quantizer.prune_model(pruning_ratio=0.3)
        models['pruned'] = pruned_model
        
        save_path = os.path.join(args.output_dir, f'{args.model_type}_pruned.pth')
        quantizer.save_optimized_model(pruned_model, save_path, 'pruning')
    
    # Benchmark models
    benchmark_results = quantizer.benchmark_models(models, test_input)
    
    # Save benchmark results
    results_path = os.path.join(args.output_dir, 'optimization_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(benchmark_results, f, default_flow_style=False)
    
    print(f"Optimization complete. Results saved to {results_path}")


if __name__ == '__main__':
    main()