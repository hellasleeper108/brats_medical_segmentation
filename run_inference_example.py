#!/usr/bin/env python3
"""
Example script demonstrating how to use the trained BraTS 2021 model for inference.
This script shows various usage patterns for the 85.51% Dice score model.
"""

import os
import sys
import time
import numpy as np
import nibabel as nib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.predictor import MedicalImagePredictor


def example_single_prediction():
    """Example: Single image prediction"""
    print("=" * 60)
    print("EXAMPLE 1: Single Image Prediction")
    print("=" * 60)
    
    # Initialize predictor with the best model
    predictor = MedicalImagePredictor(
        model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
        model_type='unet'
    )
    
    # Print model information
    model_info = predictor.get_model_info()
    print(f"Model: {model_info['model_type']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Size: {model_info['model_size_mb']:.2f} MB")
    print(f"Device: {model_info['device']}")
    print()
    
    # Example with synthetic data (replace with real image path)
    test_image_path = 'data/enhanced_synthetic/test/subject_046.nii.gz'
    
    if os.path.exists(test_image_path):
        print(f"Processing: {test_image_path}")
        
        # Perform prediction
        prediction, metrics, processing_time = predictor.predict(test_image_path)
        
        # Display results
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Classes found: {metrics['num_classes_predicted']}")
        print(f"GPU memory used: {metrics.get('gpu_memory_used', 'N/A')}")
        
        # Show class distribution
        print("\nClass Distribution:")
        for class_name, info in metrics['class_distribution'].items():
            print(f"  {class_name}: {info['voxels']:,} voxels ({info['percentage']:.2f}%)")
        
        # Save prediction
        os.makedirs('example_predictions', exist_ok=True)
        output_path = 'example_predictions/subject_046_prediction.nii.gz'
        
        # Load original image to preserve header
        original_img = nib.load(test_image_path)
        pred_img = nib.Nifti1Image(prediction, original_img.affine, original_img.header)
        nib.save(pred_img, output_path)
        
        print(f"Prediction saved to: {output_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please provide a valid NIfTI image path")
    
    print()


def example_tta_prediction():
    """Example: Test Time Augmentation for higher accuracy"""
    print("=" * 60)
    print("EXAMPLE 2: Test Time Augmentation (TTA)")
    print("=" * 60)
    
    predictor = MedicalImagePredictor(
        model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
        model_type='unet'
    )
    
    test_image_path = 'data/enhanced_synthetic/test/subject_047.nii.gz'
    
    if os.path.exists(test_image_path):
        print(f"Processing with TTA: {test_image_path}")
        
        # Compare regular vs TTA prediction
        start_time = time.time()
        prediction_regular, metrics_regular, time_regular = predictor.predict(test_image_path)
        
        prediction_tta, metrics_tta, time_tta = predictor.predict_with_tta(
            test_image_path,
            tta_transforms=['flip_x', 'flip_y', 'flip_z']
        )
        
        print(f"\nRegular prediction: {time_regular:.2f}s")
        print(f"TTA prediction: {time_tta:.2f}s")
        print(f"TTA overhead: {time_tta - time_regular:.2f}s")
        
        # Compare class distributions
        print("\nClass Distribution Comparison:")
        print("Regular vs TTA:")
        for class_name in metrics_regular['class_distribution']:
            regular_pct = metrics_regular['class_distribution'][class_name]['percentage']
            tta_pct = metrics_tta['class_distribution'][class_name]['percentage']
            print(f"  {class_name}: {regular_pct:.2f}% vs {tta_pct:.2f}%")
        
        # Save both predictions
        os.makedirs('example_predictions', exist_ok=True)
        
        original_img = nib.load(test_image_path)
        
        # Regular prediction
        pred_img_regular = nib.Nifti1Image(prediction_regular, original_img.affine, original_img.header)
        nib.save(pred_img_regular, 'example_predictions/subject_047_regular.nii.gz')
        
        # TTA prediction
        pred_img_tta = nib.Nifti1Image(prediction_tta, original_img.affine, original_img.header)
        nib.save(pred_img_tta, 'example_predictions/subject_047_tta.nii.gz')
        
        print("Predictions saved for comparison")
    else:
        print(f"Test image not found: {test_image_path}")
    
    print()


def example_batch_processing():
    """Example: Batch processing multiple images"""
    print("=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    predictor = MedicalImagePredictor(
        model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
        model_type='unet'
    )
    
    # Find test images
    test_dir = 'data/enhanced_synthetic/test'
    if os.path.exists(test_dir):
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                     if f.endswith('.nii.gz') and not f.endswith('_seg.nii.gz')]
        
        # Limit to first 3 files for demo
        test_files = test_files[:3]
        
        if test_files:
            print(f"Processing {len(test_files)} images in batch...")
            
            # Batch processing
            results = predictor.batch_predict(test_files, output_dir='example_predictions/batch/')
            
            # Summary
            successful = sum(1 for r in results if r['success'])
            total_time = sum(r.get('processing_time', 0) for r in results if r['success'])
            
            print(f"\nBatch Processing Results:")
            print(f"Total images: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/successful:.2f}s" if successful > 0 else "N/A")
            
            # Show individual results
            print("\nIndividual Results:")
            for result in results:
                if result['success']:
                    filename = os.path.basename(result['input_path'])
                    time_taken = result['processing_time']
                    print(f"  ✅ {filename}: {time_taken:.2f}s")
                else:
                    filename = os.path.basename(result['input_path'])
                    error = result['error']
                    print(f"  ❌ {filename}: {error}")
        else:
            print("No test images found for batch processing")
    else:
        print(f"Test directory not found: {test_dir}")
    
    print()


def example_performance_analysis():
    """Example: Detailed performance analysis"""
    print("=" * 60)
    print("EXAMPLE 4: Performance Analysis")
    print("=" * 60)
    
    predictor = MedicalImagePredictor(
        model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
        model_type='unet'
    )
    
    test_image_path = 'data/enhanced_synthetic/test/subject_048.nii.gz'
    
    if os.path.exists(test_image_path):
        print(f"Analyzing performance on: {test_image_path}")
        
        # Multiple runs for timing analysis
        times = []
        for i in range(5):
            _, _, processing_time = predictor.predict(test_image_path)
            times.append(processing_time)
            print(f"  Run {i+1}: {processing_time:.3f}s")
        
        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\nTiming Statistics (5 runs):")
        print(f"  Mean: {mean_time:.3f}s ± {std_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        print(f"  Coefficient of Variation: {(std_time/mean_time)*100:.1f}%")
        
        # Memory analysis (if CUDA available)
        if predictor.device.type == 'cuda':
            import torch
            torch.cuda.reset_peak_memory_stats()
            
            prediction, metrics, _ = predictor.predict(test_image_path)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"\nMemory Usage:")
            print(f"  Peak GPU memory: {peak_memory:.2f} GB")
            print(f"  Current GPU memory: {metrics['gpu_memory_used']:.2f} GB")
            print(f"  GPU memory cached: {metrics['gpu_memory_cached']:.2f} GB")
        
        # Confidence analysis
        prediction, metrics, _ = predictor.predict(test_image_path)
        confidence = metrics['confidence']
        
        print(f"\nPrediction Confidence:")
        print(f"  Mean confidence: {confidence['mean']:.3f}")
        print(f"  Std confidence: {confidence['std']:.3f}")
        print(f"  Min confidence: {confidence['min']:.3f}")
        print(f"  Max confidence: {confidence['max']:.3f}")
        
    else:
        print(f"Test image not found: {test_image_path}")
    
    print()


def main():
    """Run all examples"""
    print("🏥 BraTS 2021 Model Inference Examples")
    print("Model Performance: 85.51% Dice Score")
    print()
    
    # Check if model exists
    model_path = 'checkpoints_brats2021_rescue/best_model_rescue.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using: python train_brats2021_rescue.py")
        return 1
    
    try:
        # Run examples
        example_single_prediction()
        example_tta_prediction()
        example_batch_processing()
        example_performance_analysis()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("Check 'example_predictions/' directory for output files.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)