#!/usr/bin/env python3
"""
Move the entire AI medical segmentation project to X:/Projects/braTS
"""

import os
import shutil
import sys
from pathlib import Path

def move_project():
    source_dir = Path("C:/Projects/aimedis")
    dest_dir = Path("X:/Projects/brats_medical_segmentation")
    
    print("Moving AI Medical Segmentation Project")
    print("=" * 50)
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    # Check if source exists
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return False
    
    # Create destination parent directory
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle existing destination
    if dest_dir.exists():
        print(f"WARNING: Destination exists: {dest_dir}")
        backup_name = f"{dest_dir}_backup"
        if Path(backup_name).exists():
            print(f"Removing old backup: {backup_name}")
            shutil.rmtree(backup_name)
        print(f"Moving existing to backup: {backup_name}")
        shutil.move(str(dest_dir), backup_name)
    
    try:
        print("Copying project files...")
        shutil.copytree(source_dir, dest_dir)
        
        print("SUCCESS: Project moved successfully!")
        print(f"New location: {dest_dir}")
        
        # List key files to verify
        print("\nKey files at new location:")
        key_files = [
            "README.md",
            "requirements.txt", 
            "train_enhanced.py",
            "setup_tcia_brats.py",
            "BraTS_Download_Guide.md",
            "models/unet.py",
            "web_app/app.py"
        ]
        
        for file_path in key_files:
            full_path = dest_dir / file_path
            if full_path.exists():
                print(f"  OK: {file_path}")
            else:
                print(f"  MISSING: {file_path}")
        
        print(f"\nNext steps:")
        print(f"1. cd X:\\Projects\\brats_medical_segmentation")
        print(f"2. Download BraTS dataset from TCIA")
        print(f"3. python setup_tcia_brats.py --input_dir <downloaded_data>")
        print(f"4. python train_enhanced.py")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error moving project: {e}")
        return False

if __name__ == "__main__":
    success = move_project()
    if not success:
        sys.exit(1)