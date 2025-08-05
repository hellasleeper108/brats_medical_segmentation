import os
import io
import uuid
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet3D, VisionTransformer3D
from data.transforms import get_inference_transforms, get_post_transforms
from inference.predictor import MedicalImagePredictor
from utils.visualization import create_3d_visualization, create_slice_comparison

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables
predictor = None
model_config = None

def load_model_config(config_path='configs/config.yaml'):
    """Load model configuration."""
    global model_config
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

def initialize_predictor(model_path, model_type='unet'):
    """Initialize the medical image predictor."""
    global predictor
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    try:
        predictor = MedicalImagePredictor(
            model_path=model_path,
            model_type=model_type,
            config=model_config
        )
        print(f"Predictor initialized successfully with {model_type} model")
        return True
    except Exception as e:
        print(f"Failed to initialize predictor: {str(e)}")
        return False

def allowed_file(filename):
    """Check if uploaded file is allowed."""
    allowed_extensions = {'.nii', '.nii.gz', '.dcm', '.mha', '.mhd'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'predictor_loaded': predictor is not None,
        'model_type': model_config.get('model', {}).get('name', 'unknown') if model_config else 'unknown'
    }
    return jsonify(status)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Supported: .nii, .nii.gz, .dcm, .mha, .mhd'}), 400
    
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = ''.join(Path(filename).suffixes)
        unique_filename = f"{file_id}{file_extension}"
        
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        # Load and validate image
        try:
            img = nib.load(upload_path)
            img_data = img.get_fdata()
            
            # Basic validation
            if len(img_data.shape) != 3:
                raise ValueError("Image must be 3D")
            
            image_info = {
                'shape': img_data.shape,
                'spacing': img.header.get_zooms()[:3],
                'orientation': nib.aff2axcodes(img.affine),
                'dtype': str(img_data.dtype),
                'min_value': float(np.min(img_data)),
                'max_value': float(np.max(img_data)),
                'mean_value': float(np.mean(img_data))
            }
            
        except Exception as e:
            os.remove(upload_path)  # Clean up
            return jsonify({'error': f'Invalid medical image: {str(e)}'}), 400
        
        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'image_info': image_info,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform segmentation prediction."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'file_id' not in data:
        return jsonify({'error': 'File ID required'}), 400
    
    file_id = data['file_id']
    
    # Find uploaded file
    upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                   if f.startswith(file_id)]
    
    if not upload_files:
        return jsonify({'error': 'File not found'}), 404
    
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
    
    try:
        # Perform prediction
        prediction, metrics, processing_time = predictor.predict(upload_path)
        
        # Save prediction
        result_filename = f"{file_id}_prediction.nii.gz"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Create NIfTI file for prediction
        original_img = nib.load(upload_path)
        pred_img = nib.Nifti1Image(prediction, original_img.affine, original_img.header)
        nib.save(pred_img, result_path)
        
        # Generate visualizations
        visualization_data = create_visualization_data(upload_path, result_path, file_id)
        
        return jsonify({
            'file_id': file_id,
            'prediction_file': result_filename,
            'metrics': metrics,
            'processing_time': processing_time,
            'visualizations': visualization_data,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/visualize/<file_id>')
def get_visualization(file_id):
    """Get visualization data for a prediction."""
    try:
        # Find original and prediction files
        upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                       if f.startswith(file_id)]
        result_files = [f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                       if f.startswith(file_id) and 'prediction' in f]
        
        if not upload_files or not result_files:
            return jsonify({'error': 'Files not found'}), 404
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_files[0])
        
        visualization_data = create_visualization_data(upload_path, result_path, file_id)
        
        return jsonify(visualization_data)
        
    except Exception as e:
        return jsonify({'error': f'Visualization failed: {str(e)}'}), 500

@app.route('/api/slice/<file_id>/<int:slice_idx>/<axis>')
def get_slice_image(file_id, slice_idx, axis):
    """Get a specific slice image."""
    try:
        # Find files
        upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                       if f.startswith(file_id)]
        result_files = [f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                       if f.startswith(file_id) and 'prediction' in f]
        
        if not upload_files or not result_files:
            return jsonify({'error': 'Files not found'}), 404
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_files[0])
        
        # Load images
        original_img = nib.load(upload_path).get_fdata()
        prediction_img = nib.load(result_path).get_fdata()
        
        # Get slice based on axis
        axis_map = {'axial': 2, 'sagittal': 0, 'coronal': 1}
        if axis not in axis_map:
            return jsonify({'error': 'Invalid axis'}), 400
        
        ax = axis_map[axis]
        
        if slice_idx < 0 or slice_idx >= original_img.shape[ax]:
            return jsonify({'error': 'Invalid slice index'}), 400
        
        # Extract slices
        if ax == 0:
            orig_slice = original_img[slice_idx, :, :]
            pred_slice = prediction_img[slice_idx, :, :]
        elif ax == 1:
            orig_slice = original_img[:, slice_idx, :]
            pred_slice = prediction_img[:, slice_idx, :]
        else:
            orig_slice = original_img[:, :, slice_idx]
            pred_slice = prediction_img[:, :, slice_idx]
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Prediction overlay
        axes[1].imshow(orig_slice, cmap='gray')
        axes[1].imshow(pred_slice, cmap='jet', alpha=0.5)
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')
        
        # Prediction only
        axes[2].imshow(pred_slice, cmap='jet')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': f'Slice visualization failed: {str(e)}'}), 500

@app.route('/api/download/<file_id>/<file_type>')
def download_file(file_id, file_type):
    """Download original or prediction file."""
    try:
        if file_type == 'original':
            folder = app.config['UPLOAD_FOLDER']
            files = [f for f in os.listdir(folder) if f.startswith(file_id)]
        elif file_type == 'prediction':
            folder = app.config['RESULTS_FOLDER']
            files = [f for f in os.listdir(folder) 
                    if f.startswith(file_id) and 'prediction' in f]
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        if not files:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = os.path.join(folder, files[0])
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

def create_visualization_data(original_path, prediction_path, file_id):
    """Create visualization data for the web interface."""
    try:
        # Load images
        original_img = nib.load(original_path).get_fdata()
        prediction_img = nib.load(prediction_path).get_fdata()
        
        # Calculate slice counts for each axis
        shape = original_img.shape
        
        # Find representative slices (middle slices with content)
        def find_best_slice(img, axis):
            mid = img.shape[axis] // 2
            # Look for slice with reasonable content around middle
            for offset in range(0, min(img.shape[axis] // 4, 20)):
                for direction in [1, -1]:
                    slice_idx = mid + direction * offset
                    if 0 <= slice_idx < img.shape[axis]:
                        if axis == 0:
                            slice_data = img[slice_idx, :, :]
                        elif axis == 1:
                            slice_data = img[:, slice_idx, :]
                        else:
                            slice_data = img[:, :, slice_idx]
                        
                        if np.std(slice_data) > np.std(img) * 0.1:  # Has reasonable variation
                            return slice_idx
            return mid
        
        best_slices = {
            'axial': find_best_slice(original_img, 2),
            'sagittal': find_best_slice(original_img, 0),
            'coronal': find_best_slice(original_img, 1)
        }
        
        return {
            'shape': shape,
            'best_slices': best_slices,
            'slice_counts': {
                'axial': shape[2],
                'sagittal': shape[0],
                'coronal': shape[1]
            },
            'file_id': file_id
        }
        
    except Exception as e:
        raise Exception(f"Visualization data creation failed: {str(e)}")

# Initialize the application
if __name__ == '__main__':
    # Load configuration
    config_path = 'configs/config.yaml'
    if os.path.exists(config_path):
        load_model_config(config_path)
    
    # Try to load the best model
    model_path = 'checkpoints/best_model.pth'
    model_type = 'unet'  # Default to U-Net
    
    if model_config:
        model_type = model_config.get('model', {}).get('name', 'unet')
    
    if os.path.exists(model_path):
        success = initialize_predictor(model_path, model_type)
        if not success:
            print("Warning: Could not load model. Prediction functionality will be limited.")
    else:
        print("Warning: No trained model found. Please train a model first.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)