#!/usr/bin/env python3
"""
Setup script for AI-Powered Medical Image Segmentation system.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, check=True):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"Python version: {sys.version}")
    return True


def check_cuda_availability():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print("CUDA not available - will use CPU")
            return False
    except ImportError:
        print("PyTorch not installed yet - CUDA check will be done after installation")
        return None


def install_dependencies(gpu=True):
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Base requirements
    base_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "monai>=1.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "nibabel>=4.0.0",
        "SimpleITK>=2.2.0",
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.8.0",
        "einops>=0.6.0",
        "transformers>=4.20.0",
        "timm>=0.6.0",
        "pyyaml>=6.0.0"
    ]
    
    # Install packages
    for package in base_packages:
        if not run_command(f"pip install {package}"):
            print(f"Failed to install {package}")
            return False
    
    # Install CUDA-specific packages if GPU is available
    if gpu:
        cuda_packages = [
            "torch[cuda]",
            "torchvision[cuda]"
        ]
        for package in cuda_packages:
            run_command(f"pip install {package}", check=False)
    
    print("Dependencies installed successfully!")
    return True


def setup_directories():
    """Create necessary directories."""
    print("Setting up directory structure...")
    
    directories = [
        "data",
        "checkpoints", 
        "logs",
        "predictions",
        "visualizations",
        "optimized_models",
        "uploads",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True


def download_sample_data():
    """Download sample data for testing."""
    print("Setting up sample data...")
    
    # Run the dataset download script
    if not run_command("python data/download_datasets.py --dataset samples"):
        print("Failed to download sample data")
        return False
    
    print("Sample data setup complete!")
    return True


def test_installation():
    """Test the installation."""
    print("Testing installation...")
    
    try:
        # Test imports
        import torch
        import monai
        import nibabel
        import flask
        print("✓ All required packages imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - using CPU")
        
        # Test model creation
        from models import UNet3D, VisionTransformer3D
        
        unet = UNet3D(in_channels=1, num_classes=4)
        print("✓ U-Net model created successfully")
        
        vit = VisionTransformer3D(img_size=(128, 128, 128), in_channels=1, num_classes=4)
        print("✓ Vision Transformer model created successfully")
        
        # Test data loading
        from data.transforms import get_transforms
        transforms = get_transforms('train')
        print("✓ Data transforms created successfully")
        
        print("✅ Installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def create_quick_start_script():
    """Create a quick start script."""
    script_content = '''#!/usr/bin/env python3
"""
Quick start script for medical image segmentation.
"""

import os
import sys

def main():
    print("🏥 AI-Powered Medical Image Segmentation")
    print("=" * 50)
    
    while True:
        print("\\nWhat would you like to do?")
        print("1. Train a model")
        print("2. Run inference on an image")
        print("3. Start web application")
        print("4. Download datasets")
        print("5. Optimize model for deployment")
        print("6. Exit")
        
        choice = input("\\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\\nStarting training...")
            print("Command: python training/train.py --config configs/config.yaml")
            os.system("python training/train.py --config configs/config.yaml")
            
        elif choice == '2':
            image_path = input("Enter path to medical image: ").strip()
            if os.path.exists(image_path):
                print(f"\\nRunning inference on {image_path}")
                print("Command: python inference/run_inference.py --input", image_path)
                os.system(f"python inference/run_inference.py --input {image_path}")
            else:
                print("File not found!")
                
        elif choice == '3':
            print("\\nStarting web application...")
            print("The web app will be available at http://localhost:5000")
            os.system("python web_app/app.py")
            
        elif choice == '4':
            print("\\nDownloading datasets...")
            os.system("python data/download_datasets.py --dataset all")
            
        elif choice == '5':
            model_path = input("Enter path to trained model: ").strip()
            if os.path.exists(model_path):
                print(f"\\nOptimizing model {model_path}")
                os.system(f"python deployment/quantization.py --model_path {model_path}")
            else:
                print("Model file not found!")
                
        elif choice == '6':
            print("Goodbye! 👋")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
'''
    
    with open('quick_start.py', 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if sys.platform != 'win32':
        os.chmod('quick_start.py', 0o755)
    
    print("Created quick_start.py script")


def create_docker_files():
    """Create Docker configuration files."""
    
    # Dockerfile
    dockerfile_content = '''FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data checkpoints logs predictions visualizations uploads results

# Expose port for web application
EXPOSE 5000

# Default command
CMD ["python", "web_app/app.py"]
'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    # Docker Compose
    compose_content = '''version: '3.8'

services:
  aimedis:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./predictions:/app/predictions
      - ./uploads:/app/uploads
      - ./results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  training:
    build: .
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: python training/train.py --config configs/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("Created Docker configuration files")


def main():
    parser = argparse.ArgumentParser(description='Setup AI Medical Image Segmentation system')
    parser.add_argument('--no-gpu', action='store_true', help='Skip GPU-specific setup')
    parser.add_argument('--no-data', action='store_true', help='Skip sample data download')
    parser.add_argument('--docker', action='store_true', help='Create Docker files')
    parser.add_argument('--test-only', action='store_true', help='Only run installation test')
    
    args = parser.parse_args()
    
    print("🏥 AI-Powered Medical Image Segmentation Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    if args.test_only:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    # Check CUDA
    gpu_available = check_cuda_availability()
    use_gpu = gpu_available and not args.no_gpu
    
    # Install dependencies
    if not install_dependencies(gpu=use_gpu):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        print("❌ Failed to setup directories")
        sys.exit(1)
    
    # Download sample data
    if not args.no_data:
        if not download_sample_data():
            print("⚠ Failed to download sample data - continuing anyway")
    
    # Create helper scripts
    create_quick_start_script()
    
    # Create Docker files if requested
    if args.docker:
        create_docker_files()
    
    # Test installation
    if test_installation():
        print("\\n🎉 Setup completed successfully!")
        print("\\nNext steps:")
        print("1. Run 'python quick_start.py' for interactive menu")
        print("2. Or train a model: 'python training/train.py'")
        print("3. Or start web app: 'python web_app/app.py'")
        print("\\n📚 Check README.md for detailed documentation")
    else:
        print("\\n❌ Setup completed with errors")
        print("Please check the error messages above")


if __name__ == '__main__':
    main()