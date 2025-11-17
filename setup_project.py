"""
Hospital Readmission Predictor - Fixed Windows Setup Script
For Shilpa's MSc Data Science Project
"""
import subprocess
import sys
import os
from pathlib import Path
import time

def print_header():
    """Print project header"""
    print("=" * 70)
    print("    HOSPITAL READMISSION PREDICTOR")
    print("    MSc Data Science Dissertation Project")
    print("    Student: Shilpa Sunil (001422153)")
    print("    Supervisor: Mr. Chan Sorayudh Chanthan")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"[OK] Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_location():
    """Verify we're in the correct project location"""
    print("\nChecking project location...")
    
    current_path = Path.cwd()
    print(f"Current directory: {current_path}")
    
    if not str(current_path).endswith("hospital_readmission_predictor"):
        expected_path = Path(r"D:\Rkr\Masters\Course Module\Project\Shilpa\Code\hospital_readmission_predictor")
        print(f"[WARNING] Expected to be in: {expected_path}")
        print("Please navigate to the correct directory first")
        return False
    
    print("[OK] Correct project location")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("\nSetting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("[OK] Virtual environment already exists")
        return True
    
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("[OK] Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to create virtual environment")
        return False

def install_requirements():
    """Install project requirements manually"""
    print("\nInstalling requirements...")
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("[OK] Virtual environment is active")
        pip_cmd = "pip"
        python_cmd = "python"
    else:
        print("[WARNING] Virtual environment not active, using venv paths")
        if os.name == 'nt':  # Windows
            pip_cmd = str(Path("venv") / "Scripts" / "pip.exe")
            python_cmd = str(Path("venv") / "Scripts" / "python.exe")
        else:
            pip_cmd = str(Path("venv") / "bin" / "pip")
            python_cmd = str(Path("venv") / "bin" / "python")
    
    # Core packages to install
    packages = [
        "pandas==2.0.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "streamlit==1.25.0",
        "joblib==1.3.1"
    ]
    
    try:
        # Upgrade pip first
        subprocess.check_call([python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
        print("[OK] pip upgraded")
        
        # Install packages one by one
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([python_cmd, "-m", "pip", "install", package])
            print(f"[OK] {package} installed")
        
        print("[OK] All requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install requirements: {e}")
        return False

def create_directories():
    """Create all required directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data", "data/raw", "data/processed", "data/demo",
        "src", "src/data_processing", "src/models", "src/visualization",
        "models", "models/trained",
        "app", "app/pages",
        "notebooks", "tests", "docs",
        "outputs", "outputs/figures", "outputs/reports"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] Created {len(directories)} directories")
    return True

def create_init_files():
    """Create __init__.py files for Python packages"""
    print("\nCreating package files...")
    
    init_locations = [
        "src/__init__.py",
        "src/data_processing/__init__.py",
        "src/models/__init__.py",
        "src/visualization/__init__.py",
        "app/__init__.py"
    ]
    
    for init_file in init_locations:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.write_text('"""Package initialization file"""', encoding='utf-8')
    
    print(f"[OK] Created {len(init_locations)} package files")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\nTesting package imports...")
    
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('streamlit', 'st')
    ]
    
    failed_imports = []
    
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [ERROR] {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n[WARNING] Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("[OK] All packages imported successfully")
    return True

def test_config():
    """Test the configuration file"""
    print("\nTesting configuration...")
    
    try:
        sys.path.append("src")
        from config import PROJECT_ROOT, DATA_DIR, MODELS_DIR
        
        print(f"  [OK] Project root: {PROJECT_ROOT}")
        print(f"  [OK] Data directory: {DATA_DIR}")
        print(f"  [OK] Models directory: {MODELS_DIR}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Configuration error: {e}")
        return False

def test_data_loader():
    """Test the data loader"""
    print("\nTesting data loader...")
    
    try:
        sys.path.append("src")
        from data_processing.data_loader import load_demo_dataset
        
        print("  Creating sample dataset...")
        df = load_demo_dataset()
        
        print(f"  [OK] Dataset shape: {df.shape}")
        print(f"  [OK] Readmission rate: {df['readmission_30_day'].mean():.1%}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Data loader error: {e}")
        return False

def create_sample_notebook():
    """Create a sample Jupyter notebook without emojis"""
    print("\nCreating sample notebook...")
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hospital Readmission Prediction - Quick Start\\n",
    "## MSc Data Science Project - Shilpa Sunil\\n",
    "\\n",
    "This notebook demonstrates the basic functionality of the project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import sys\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "from data_processing.data_loader import load_demo_dataset\\n",
    "\\n",
    "print('Hospital Readmission Prediction Project')\\n",
    "print('Student: Shilpa Sunil (001422153)')\\n",
    "print('Supervisor: Mr. Chan Sorayudh Chanthan')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load demo dataset\\n",
    "df = load_demo_dataset()\\n",
    "print(f'Dataset shape: {df.shape}')\\n",
    "print(f'Readmission rate: {df[\"readmission_30_day\"].mean():.1%}')\\n",
    "df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    try:
        notebook_path = Path("notebooks") / "00_quick_start.ipynb"
        notebook_path.write_text(notebook_content, encoding='utf-8')
        print(f"[OK] Created notebook: {notebook_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create notebook: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("PROJECT SETUP COMPLETE!")
    print("=" * 60)
    
    print("\nNEXT STEPS:")
    print("=" * 30)
    print("1. Explore the data:")
    print("   jupyter notebook notebooks/00_quick_start.ipynb")
    print()
    print("2. Test data processing:")
    print("   python src/data_processing/data_loader.py")
    print()
    print("3. Move to Phase 2:")
    print("   - Feature engineering")
    print("   - Machine learning models")
    print("   - Streamlit web application")
    print()
    print("4. Your MIMIC-III data (if available):")
    print("   Place CSV files in: data/raw/")
    print()
    print("PROJECT TIMELINE:")
    print("  Week 8-9:  Data preprocessing & basic models")
    print("  Week 10-11: Advanced models & time-series")
    print("  Week 12-13: Web application & evaluation")
    print("  Week 14-16: Optimization & documentation")
    print()
    print("Contact supervisor for guidance:")
    print("   Mr. Chan Sorayudh Chanthan")

def main():
    """Main setup function"""
    print_header()
    
    # Run setup steps
    setup_steps = [
        ("Python Version", check_python_version),
        ("Project Location", check_location),
        ("Virtual Environment", create_virtual_environment),
        ("Package Installation", install_requirements),
        ("Directory Structure", create_directories),
        ("Package Files", create_init_files),
        ("Import Testing", test_imports),
        ("Configuration", test_config),
        ("Data Loader", test_data_loader),
        ("Sample Notebook", create_sample_notebook)
    ]
    
    results = []
    
    for step_name, step_function in setup_steps:
        print(f"\n{'=' * 60}")
        print(f"STEP: {step_name}")
        print(f"{'=' * 60}")
        
        try:
            result = step_function()
            results.append(result)
            
            if result:
                print(f"[OK] {step_name} completed successfully")
            else:
                print(f"[ERROR] {step_name} failed")
                
        except Exception as e:
            print(f"[ERROR] {step_name} failed with error: {e}")
            results.append(False)
        
        time.sleep(1)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SETUP SUMMARY")
    print(f"{'=' * 60}")
    
    success_count = sum(results)
    total_count = len(results)
    
    for i, (step_name, _) in enumerate(setup_steps):
        status = "[OK]" if results[i] else "[ERROR]"
        print(f"{status} {step_name}")
    
    print(f"\nSuccess Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print_next_steps()
    else:
        print("\n[WARNING] Some setup steps failed. Please check the errors above.")

if __name__ == "__main__":
    main()