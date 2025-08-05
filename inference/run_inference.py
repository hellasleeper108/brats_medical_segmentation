#!/usr/bin/env python3
"""
Command-line inference script for medical image segmentation.
"""

import os
import sys
import argparse
import time
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import MedicalImagePredictor
from utils.visualization import create_segmentation_summary
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Inference')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input medical image')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: predictions/)')
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='unet',
                       choices=['unet', 'vit'],
                       help='Type of model')
    parser.add_argument('--config', '-c', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--tta', action='store_true',
                       help='Use test time augmentation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    parser.add_argument('--batch', type=str, default=None,
                       help='Directory for batch processing')
    
    args = parser.parse_args()
    
    # Check if input exists
    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Error: Batch directory not found: {args.batch}")
            return 1
        input_files = [os.path.join(args.batch, f) for f in os.listdir(args.batch)
                      if f.lower().endswith(('.nii', '.nii.gz'))]
        if not input_files:
            print(f"Error: No NIfTI files found in {args.batch}")
            return 1
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1
        input_files = [args.input]
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train a model first or provide a valid model path")
        return 1
    
    # Set output directory
    if args.output is None:
        args.output = 'predictions'
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file not found: {args.config}")
        config = None
    
    print("🏥 Medical Image Segmentation Inference")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Input files: {len(input_files)}")
    print(f"Output directory: {args.output}")
    print(f"TTA enabled: {args.tta}")
    print()
    
    try:
        # Initialize predictor
        print("Loading model...")
        predictor = MedicalImagePredictor(
            model_path=args.model,
            model_type=args.model_type,
            config=config
        )
        
        # Print model info
        model_info = predictor.get_model_info()
        print(f"Model parameters: {model_info['total_parameters']:,}")
        print(f"Model size: {model_info['model_size_mb']:.2f} MB")
        print()
        
        # Process files
        total_time = 0
        results = []
        
        for i, input_file in enumerate(input_files):
            print(f"Processing {i+1}/{len(input_files)}: {os.path.basename(input_file)}")
            
            start_time = time.time()
            
            try:
                # Run prediction
                if args.tta:
                    prediction, metrics, processing_time = predictor.predict_with_tta(input_file)
                else:
                    prediction, metrics, processing_time = predictor.predict(input_file)
                
                total_time += processing_time
                
                # Save prediction
                base_name = Path(input_file).stem
                if base_name.endswith('.nii'):
                    base_name = base_name[:-4]
                
                output_file = os.path.join(args.output, f"{base_name}_prediction.nii.gz")
                
                # Load original image to get header info
                original_img = nib.load(input_file)
                pred_img = nib.Nifti1Image(prediction, original_img.affine, original_img.header)
                nib.save(pred_img, output_file)
                
                # Print results
                print(f"  Processing time: {processing_time:.2f}s")
                print(f"  Prediction shape: {prediction.shape}")
                print(f"  Classes found: {len(np.unique(prediction))}")
                print(f"  Saved to: {output_file}")
                
                # Create visualization if requested
                if args.visualize:
                    vis_dir = os.path.join(args.output, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    try:
                        vis_files = create_segmentation_summary(
                            input_file, output_file, vis_dir, base_name
                        )
                        print(f"  Visualizations saved to: {vis_dir}")
                    except Exception as e:
                        print(f"  Warning: Visualization failed: {e}")
                
                results.append({
                    'input_file': input_file,
                    'output_file': output_file,
                    'processing_time': processing_time,
                    'metrics': metrics,
                    'success': True
                })
                
                print(f"  ✅ Success")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'input_file': input_file,
                    'error': str(e),
                    'success': False
                })
            
            print()
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print("=" * 50)
        print("SUMMARY")
        print(f"Total files processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total processing time: {total_time:.2f}s")
        if successful > 0:
            print(f"Average time per file: {total_time/successful:.2f}s")
        
        # Save detailed results
        results_file = os.path.join(args.output, 'inference_results.yaml')
        with open(results_file, 'w') as f:
            yaml.dump({
                'summary': {
                    'total_files': len(results),
                    'successful': successful,
                    'failed': failed,
                    'total_time': total_time,
                    'average_time': total_time/successful if successful > 0 else 0
                },
                'results': results
            }, f, default_flow_style=False)
        
        print(f"Detailed results saved to: {results_file}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())