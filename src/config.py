
import os
from pathlib import Path

# Project paths - Windows specific
PROJECT_ROOT = Path(r"D:\Rkr\Masters\Course Module\Project\Shilpa\Code\hospital_readmission_predictor")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files (MIMIC-III)
MIMIC_ADMISSIONS = RAW_DATA_DIR / "ADMISSIONS.csv"
MIMIC_PATIENTS = RAW_DATA_DIR / "PATIENTS.csv"
MIMIC_DIAGNOSES = RAW_DATA_DIR / "DIAGNOSES_ICD.csv"
MIMIC_PROCEDURES = RAW_DATA_DIR / "PROCEDURES_ICD.csv"
MIMIC_LABEVENTS = RAW_DATA_DIR / "LABEVENTS.csv"
MIMIC_CHARTEVENTS = RAW_DATA_DIR / "CHARTEVENTS.csv"

# Processed data files
PROCESSED_FEATURES = PROCESSED_DATA_DIR / "patient_features.csv"
PROCESSED_TIMESERIES = PROCESSED_DATA_DIR / "timeseries_features.csv"
FINAL_DATASET = PROCESSED_DATA_DIR / "final_dataset.csv"
DEMO_DATASET = DEMO_DATA_DIR / "sample_patient_data.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
READMISSION_WINDOW = 30  # days
MIN_AGE = 18
MAX_AGE = 120

# Time series parameters
TIME_WINDOW = 24  # hours before discharge
MIN_MEASUREMENTS = 3  # minimum measurements per feature

# Model filenames
RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest_model.pkl"
LOGISTIC_REGRESSION_MODEL = MODELS_DIR / "logistic_regression_model.pkl"
NEURAL_NETWORK_MODEL = MODELS_DIR / "neural_network_model.h5"
LSTM_MODEL = MODELS_DIR / "lstm_model.h5"
SCALER_MODEL = MODELS_DIR / "feature_scaler.pkl"
LABEL_ENCODER_MODEL = MODELS_DIR / "label_encoder.pkl"
SHAP_EXPLAINER = MODELS_DIR / "shap_explainer.pkl"

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Hospital Readmission Predictor",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Feature categories for your model
DEMOGRAPHIC_FEATURES = [
    'age', 'gender', 'ethnicity', 'insurance'
]

CLINICAL_FEATURES = [
    'length_of_stay', 'num_diagnoses', 'num_procedures', 
    'admission_type', 'hospital_expire_flag'
]

COMORBIDITY_FEATURES = [
    'diabetes', 'hypertension', 'heart_failure', 'copd',
    'renal_failure', 'liver_disease', 'cancer'
]

EMERGENCY_FEATURES = [
    'previous_ed_visits', 'emergency_admission', 
    'admissions_last_year'
]

LAB_FEATURES = [
    'hemoglobin', 'sodium', 'potassium', 'creatinine',
    'glucose', 'white_blood_cell_count'
]

# LACE Score components (from your literature review)
LACE_COMPONENTS = {
    'length_of_stay': {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 
        'over_14': 14
    },
    'acuity': {
        'EMERGENCY': 3, 
        'URGENT': 1, 
        'ELECTIVE': 0
    },
    'comorbidities': 'charlson_comorbidity_index',  # 0-6+ points
    'ed_visits': {0: 0, 1: 1, 2: 2, 3: 3, 'over_4': 4}
}

# HOSPITAL Score components (from your literature review)
HOSPITAL_COMPONENTS = {
    'hemoglobin_low': 1,      # <12 g/dL = 1 point
    'oncology_service': 2,    # Discharge from oncology = 2 points
    'sodium_low': 1,          # <135 mEq/L = 1 point
    'procedure_index': 1,     # Any procedure during admission = 1 point
    'admission_type': 1,      # Emergency admission = 1 point
    'admissions_last_year': { # Number of admissions in past year
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 'over_5': 5
    },
    'length_of_stay': {       # Length of current stay
        'under_5': 1, 'over_5': 2
    }
}

# ICD-9 codes for common comorbidities (for your feature engineering)
ICD9_COMORBIDITIES = {
    'diabetes': ['25000', '25001', '25002', '25003'],
    'hypertension': ['4019', '4010', '4011', '4012'],
    'heart_failure': ['4280', '4281', '42821', '42831'],
    'copd': ['49121', '49320', '49390'],
    'renal_failure': ['5849', '5856', '5859'],
    'liver_disease': ['5712', '5715', '5716'],
    'cancer': ['1400', '1401', '1402', '1403']  # Various cancer codes
}

# Lab test item IDs from MIMIC-III (if you get access to real data)
MIMIC_LAB_ITEMS = {
    'hemoglobin': [50811, 51222],
    'sodium': [50824, 50983],
    'potassium': [50822, 50971],
    'creatinine': [50912],
    'glucose': [50809, 50931],
    'white_blood_cell': [51300, 51301]
}

# Create directories if they don't exist
def create_directories():
    """Create all required directories"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR,
        MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, SRC_DIR, APP_DIR, NOTEBOOKS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

# Validation function
def validate_setup():
    """Validate that all directories exist"""
    required_dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR,
        MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    else:
        print(" All directories exist")
        return True

if __name__ == "__main__":
    # Create directories when this file is run
    create_directories()
    validate_setup()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")