"""
Complete setup validation for Hospital Readmission Predictor
MSc Data Science Project - Shilpa Sunil
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

def print_test_header():
    """Print test header"""
    print("🧪" + "=" * 60 + "🧪")
    print("    COMPLETE SETUP VALIDATION TEST")
    print("    Hospital Readmission Predictor")
    print("    MSc Data Science - Shilpa Sunil")
    print("🧪" + "=" * 60 + "🧪")
    print()

def test_directory_structure():
    """Test that all required directories exist"""
    print("📁 Testing directory structure...")
    
    required_dirs = [
        "data", "data/raw", "data/processed", "data/demo",
        "src", "src/data_processing", "src/models", "src/visualization",
        "models", "models/trained",
        "app", "app/pages",
        "notebooks", "tests", "docs",
        "outputs", "outputs/figures", "outputs/reports"
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            existing_dirs.append(dir_name)
        else:
            missing_dirs.append(dir_name)
    
    print(f"  ✅ Found {len(existing_dirs)} directories")
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    
    print("  ✅ All required directories exist")
    return True

def test_core_files():
    """Test that core files exist"""
    print("\n📄 Testing core files...")
    
    required_files = [
        "requirements.txt",
        "src/config.py",
        "src/data_processing/data_loader.py",
        ".gitignore"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            existing_files.append(file_name)
        else:
            missing_files.append(file_name)
    
    print(f"  ✅ Found {len(existing_files)} core files")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All core files exist")
    return True

def test_python_imports():
    """Test Python package imports"""
    print("\n🐍 Testing Python imports...")
    
    test_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('joblib', None)
    ]
    
    failed_imports = []
    successful_imports = []
    
    for package, alias in test_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            successful_imports.append(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            failed_imports.append(package)
            print(f"  ❌ {package}: {e}")
    
    if failed_imports:
        print(f"\n  ⚠️  Failed imports: {failed_imports}")
        return False
    
    print(f"  ✅ All {len(successful_imports)} packages imported successfully")
    return True

def test_configuration():
    """Test configuration file"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, 
            RANDOM_STATE, READMISSION_WINDOW
        )
        
        print(f"  ✅ Project root: {PROJECT_ROOT}")
        print(f"  ✅ Data directory: {DATA_DIR}")
        print(f"  ✅ Models directory: {MODELS_DIR}")
        print(f"  ✅ Random state: {RANDOM_STATE}")
        print(f"  ✅ Readmission window: {READMISSION_WINDOW} days")
        
        # Test directory creation
        if hasattr(sys.modules['config'], 'create_directories'):
            from config import create_directories
            create_directories()
            print("  ✅ Directory creation function works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_data_loader():
    """Test data loading functionality"""
    print("\n📊 Testing data loader...")
    
    try:
        from data_processing.data_loader import HospitalDataProcessor, load_demo_dataset
        
        # Test data processor initialization
        processor = HospitalDataProcessor()
        print("  ✅ Data processor initialized")
        
        # Test sample data creation
        sample_admissions = processor._create_sample_admissions(100)
        print(f"  ✅ Sample admissions created: {sample_admissions.shape}")
        
        # Test demo dataset creation
        print("  📈 Creating demo dataset...")
        demo_df = load_demo_dataset()
        
        print(f"  ✅ Demo dataset shape: {demo_df.shape}")
        print(f"  ✅ Columns: {len(demo_df.columns)}")
        print(f"  ✅ Readmission rate: {demo_df['readmission_30_day'].mean():.1%}")
        
        # Validate dataset structure
        required_columns = [
            'SUBJECT_ID', 'age', 'gender', 'length_of_stay', 
            'readmission_30_day', 'emergency_admission'
        ]
        
        missing_columns = [col for col in required_columns if col not in demo_df.columns]
        
        if missing_columns:
            print(f"  ❌ Missing columns: {missing_columns}")
            return False
        
        print("  ✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"  ❌ Data loader error: {e}")
        return False

def test_basic_analytics():
    """Test basic analytics on the demo dataset"""
    print("\n📈 Testing basic analytics...")
    
    try:
        from data_processing.data_loader import load_demo_dataset
        
        df = load_demo_dataset()
        
        # Basic statistics
        stats = {
            'total_patients': df['SUBJECT_ID'].nunique(),
            'total_admissions': len(df),
            'avg_age': df['age'].mean(),
            'avg_los': df['length_of_stay'].mean(),
            'readmission_rate': df['readmission_30_day'].mean(),
            'emergency_rate': df['emergency_admission'].mean()
        }
        
        print(f"  ✅ Unique patients: {stats['total_patients']}")
        print(f"  ✅ Total admissions: {stats['total_admissions']}")
        print(f"  ✅ Average age: {stats['avg_age']:.1f} years")
        print(f"  ✅ Average LOS: {stats['avg_los']:.1f} days")
        print(f"  ✅ Readmission rate: {stats['readmission_rate']:.1%}")
        print(f"  ✅ Emergency admission rate: {stats['emergency_rate']:.1%}")
        
        # Validate reasonable ranges
        validation_checks = [
            (stats['avg_age'] > 18 and stats['avg_age'] < 100, "Age range"),
            (stats['avg_los'] > 0 and stats['avg_los'] < 50, "Length of stay range"),
            (stats['readmission_rate'] > 0 and stats['readmission_rate'] < 0.5, "Readmission rate range"),
            (stats['total_patients'] > 0, "Patient count")
        ]
        
        for check_passed, check_name in validation_checks:
            if check_passed:
                print(f"  ✅ {check_name} validation passed")
            else:
                print(f"  ❌ {check_name} validation failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Analytics error: {e}")
        return False

def test_risk_scores():
    """Test risk score calculations"""
    print("\n🎯 Testing risk score calculations...")
    
    try:
        from data_processing.data_loader import load_demo_dataset
        
        df = load_demo_dataset()
        
        # Check if risk scores exist
        if 'lace_total' in df.columns:
            lace_stats = {
                'mean': df['lace_total'].mean(),
                'min': df['lace_total'].min(),
                'max': df['lace_total'].max(),
                'high_risk': (df['lace_total'] > 10).sum()
            }
            
            print(f"  ✅ LACE score - Mean: {lace_stats['mean']:.1f}")
            print(f"  ✅ LACE score - Range: {lace_stats['min']}-{lace_stats['max']}")
            print(f"  ✅ High LACE risk patients: {lace_stats['high_risk']}")
        
        if 'hospital_score' in df.columns:
            hospital_mean = df['hospital_score'].mean()
            print(f"  ✅ HOSPITAL score - Mean: {hospital_mean:.1f}")
        
        # Test comorbidity features
        comorbidity_cols = ['diabetes', 'hypertension', 'heart_failure', 'copd']
        for col in comorbidity_cols:
            if col in df.columns:
                prevalence = df[col].mean()
                print(f"  ✅ {col.title()} prevalence: {prevalence:.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Risk score error: {e}")
        return False

def test_data_export():
    """Test data export functionality"""
    print("\n💾 Testing data export...")
    
    try:
        from data_processing.data_loader import load_demo_dataset
        from config import DEMO_DATA_DIR
        
        df = load_demo_dataset()
        
        # Test CSV export
        export_path = DEMO_DATA_DIR / "test_export.csv"
        df.to_csv(export_path, index=False)
        
        # Verify export
        if export_path.exists():
            exported_df = pd.read_csv(export_path)
            
            if len(exported_df) == len(df) and len(exported_df.columns) == len(df.columns):
                print(f"  ✅ Data exported successfully to {export_path}")
                print(f"  ✅ Export verified: {exported_df.shape}")
                
                # Clean up test file
                export_path.unlink()
                print("  ✅ Test file cleaned up")
                
                return True
            else:
                print("  ❌ Export verification failed")
                return False
        else:
            print("  ❌ Export file not created")
            return False
            
    except Exception as e:
        print(f"  ❌ Export error: {e}")
        return False

def generate_summary_report():
    """Generate a summary report"""
    print("\n📋 Generating project summary...")
    
    try:
        from data_processing.data_loader import load_demo_dataset
        from config import PROJECT_ROOT, DATA_DIR
        
        df = load_demo_dataset()
        
        summary = f"""
🏥 HOSPITAL READMISSION PREDICTOR - PROJECT SUMMARY
{'=' * 60}

📊 DATASET OVERVIEW:
  • Total Patients: {df['SUBJECT_ID'].nunique():,}
  • Total Admissions: {len(df):,}
  • 30-day Readmissions: {df['readmission_30_day'].sum():,} ({df['readmission_30_day'].mean():.1%})
  • Average Age: {df['age'].mean():.1f} years
  • Average Length of Stay: {df['length_of_stay'].mean():.1f} days

🏥 CLINICAL FEATURES:
  • Emergency Admissions: {df['emergency_admission'].mean():.1%}
  • Diabetes Patients: {df['diabetes'].mean():.1%}
  • Hypertension Patients: {df['hypertension'].mean():.1%}
  • Heart Failure Patients: {df['heart_failure'].mean():.1%}

📈 RISK SCORES:
  • Average LACE Score: {df['lace_total'].mean():.1f}
  • High Risk Patients (LACE > 10): {(df['lace_total'] > 10).sum():,}
  
📁 PROJECT STRUCTURE:
  • Project Root: {PROJECT_ROOT}
  • Data Directory: {DATA_DIR}
  • Demo Dataset: {len(df)} records ready for ML

🎯 NEXT STEPS:
  1. Feature engineering and selection
  2. Machine learning model development
  3. Time-series analysis implementation
  4. Streamlit web application
  5. Model evaluation and comparison

✅ PROJECT STATUS: Ready for Phase 2 Development
        """
        
        print(summary)
        
        # Save summary to file
        summary_path = PROJECT_ROOT / "outputs" / "project_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary)
        
        print(f"📄 Summary saved to: {summary_path}")
        return True
        
    except Exception as e:
        print(f"❌ Summary generation error: {e}")
        return False

def main():
    """Run all tests"""
    print_test_header()
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Core Files", test_core_files),
        ("Python Imports", test_python_imports),
        ("Configuration", test_configuration),
        ("Data Loader", test_data_loader),
        ("Basic Analytics", test_basic_analytics),
        ("Risk Scores", test_risk_scores),
        ("Data Export", test_data_export),
        ("Summary Report", generate_summary_report)
    ]
    
    results = []
    
    for test_name, test_function in tests:
        print(f"\n{'=' * 60}")
        print(f"🧪 RUNNING TEST: {test_name}")
        print(f"{'=' * 60}")
        
        try:
            result = test_function()
            results.append(result)
            
            if result:
                print(f"\n✅ {test_name} - PASSED")
            else:
                print(f"\n❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"\n💥 {test_name} - ERROR: {e}")
            results.append(False)
    
    # Final results
    print(f"\n{'🎯' * 20}")
    print("FINAL TEST RESULTS")
    print(f"{'🎯' * 20}")
    
    passed_tests = sum(results)
    total_tests = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    success_rate = passed_tests / total_tests * 100
    print(f"\n📊 SUCCESS RATE: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Project is ready for Phase 2: Model Development")
        print("\n📋 RECOMMENDED NEXT ACTIONS:")
        print("  1. Start Jupyter notebook: jupyter notebook")
        print("  2. Explore data: notebooks/00_quick_start.ipynb")
        print("  3. Begin feature engineering")
        print("  4. Develop baseline models")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed")
        print("🔧 Please fix issues before proceeding to Phase 2")

if __name__ == "__main__":
    main()