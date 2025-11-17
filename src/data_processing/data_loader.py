
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(r"D:\Rkr\Masters\Course Module\Project\Shilpa\Code\hospital_readmission_predictor")
sys.path.append(str(PROJECT_ROOT / "src"))

# Import config after adding to path
try:
    from config import *
except ImportError:
    print("Config not found. Using default paths...")
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    DEMO_DATA_DIR = DATA_DIR / "demo"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HospitalDataProcessor:
    """
    Main class for processing hospital readmission data
    Handles both MIMIC-III data and sample data generation
    """
    
    def __init__(self):
        self.data_files = {
            'admissions': RAW_DATA_DIR / "ADMISSIONS.csv",
            'patients': RAW_DATA_DIR / "PATIENTS.csv",
            'diagnoses': RAW_DATA_DIR / "DIAGNOSES_ICD.csv",
            'procedures': RAW_DATA_DIR / "PROCEDURES_ICD.csv",
            'labevents': RAW_DATA_DIR / "LABEVENTS.csv"
        }
        self.loaded_data = {}
        self.processed_features = None
        
        # Ensure directories exist
        for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_mimic_data(self, table_name: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load MIMIC-III data if available, otherwise create sample data
        
        Args:
            table_name: Name of the table to load
            nrows: Number of rows to load (None for all)
            
        Returns:
            DataFrame with the loaded data
        """
        if table_name not in self.data_files:
            raise ValueError(f"Unknown table: {table_name}")
        
        file_path = self.data_files[table_name]
        
        if file_path.exists():
            logger.info(f"Loading real MIMIC-III data from {file_path}")
            try:
                df = pd.read_csv(file_path, nrows=nrows)
                logger.info(f"Loaded {len(df)} rows from {table_name}")
                self.loaded_data[table_name] = df
                return df
            except Exception as e:
                logger.error(f"Error loading {table_name}: {str(e)}")
                logger.info("Falling back to sample data...")
        
        # Create sample data if MIMIC-III not available
        logger.info(f"Creating sample {table_name} data for development...")
        df = self._create_sample_data(table_name, nrows or 1000)
        self.loaded_data[table_name] = df
        return df
    
    def _create_sample_data(self, table_name: str, nrows: int) -> pd.DataFrame:
        """Create realistic sample data for development"""
        np.random.seed(42)  # For reproducible sample data
        
        if table_name == 'admissions':
            return self._create_sample_admissions(nrows)
        elif table_name == 'patients':
            return self._create_sample_patients(nrows // 3)
        elif table_name == 'diagnoses':
            return self._create_sample_diagnoses(nrows * 2)
        elif table_name == 'procedures':
            return self._create_sample_procedures(nrows)
        elif table_name == 'labevents':
            return self._create_sample_labevents(nrows * 10)
        else:
            return pd.DataFrame()
    
    def _create_sample_admissions(self, nrows: int) -> pd.DataFrame:
        """Create sample admissions data based on real MIMIC-III structure"""
        
        # Create realistic admission data
        base_date = pd.Timestamp('2008-01-01')
        num_patients = nrows // 3  # Each patient has ~3 admissions on average
        
        data = []
        hadm_id = 1
        
        for subject_id in range(1, num_patients + 1):
            # Each patient has 1-5 admissions
            num_admissions = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            last_discharge = None
            
            for admission_num in range(num_admissions):
                # Admission timing
                if last_discharge is None:
                    admit_time = base_date + pd.Timedelta(days=np.random.randint(0, 1000))
                else:
                    # Some admissions are readmissions within 30 days
                    if np.random.random() < 0.15:  # 15% chance of readmission
                        days_gap = np.random.randint(1, 30)
                    else:
                        days_gap = np.random.randint(31, 365)
                    admit_time = last_discharge + pd.Timedelta(days=days_gap)
                
                # Length of stay (realistic distribution)
                los_days = max(1, int(np.random.exponential(5)))  # Average 5 days
                discharge_time = admit_time + pd.Timedelta(days=los_days)
                
                # Admission details
                admission_type = np.random.choice(
                    ['EMERGENCY', 'ELECTIVE', 'URGENT'], 
                    p=[0.6, 0.3, 0.1]
                )
                
                # Demographics (consistent per patient)
                if admission_num == 0:  # Set demographics for first admission
                    gender = np.random.choice(['M', 'F'])
                    ethnicity = np.random.choice([
                        'WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC/LATINO', 
                        'ASIAN', 'OTHER'
                    ], p=[0.6, 0.15, 0.1, 0.1, 0.05])
                    insurance = np.random.choice([
                        'Medicare', 'Private', 'Medicaid', 'Self Pay'
                    ], p=[0.4, 0.35, 0.2, 0.05])
                
                # Death flag (small percentage)
                expire_flag = 1 if np.random.random() < 0.05 else 0
                if expire_flag:
                    discharge_time = pd.NaT
                
                admission_data = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': hadm_id,
                    'ADMITTIME': admit_time,
                    'DISCHTIME': discharge_time,
                    'ADMISSION_TYPE': admission_type,
                    'INSURANCE': insurance,
                    'ETHNICITY': ethnicity,
                    'HOSPITAL_EXPIRE_FLAG': expire_flag
                }
                
                data.append(admission_data)
                hadm_id += 1
                last_discharge = discharge_time if pd.notna(discharge_time) else None
                
                if len(data) >= nrows:
                    break
            
            if len(data) >= nrows:
                break
        
        df = pd.DataFrame(data[:nrows])
        return df
    
    def _create_sample_patients(self, nrows: int) -> pd.DataFrame:
        """Create sample patients data"""
        np.random.seed(42)
        
        # Realistic age distribution for hospital patients
        ages = np.random.choice(
            range(18, 95), 
            size=nrows, 
            p=self._get_age_distribution()
        )
        
        birth_years = [2023 - age for age in ages]
        
        data = {
            'SUBJECT_ID': range(1, nrows + 1),
            'GENDER': np.random.choice(['M', 'F'], nrows, p=[0.52, 0.48]),
            'DOB': [pd.Timestamp(f'{year}-{np.random.randint(1,13)}-{np.random.randint(1,28)}') 
                   for year in birth_years],
            'DOD': [None] * nrows  # Most patients don't die in dataset
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_diagnoses(self, nrows: int) -> pd.DataFrame:
        """Create sample diagnoses with realistic ICD-9 codes"""
        np.random.seed(42)
        
        # Common ICD-9 codes based on readmission literature
        common_diagnoses = {
            '41401': 0.08,    # Coronary atherosclerosis
            '4280': 0.12,     # Heart failure
            '49121': 0.06,    # COPD with exacerbation
            '25000': 0.10,    # Diabetes mellitus
            '4019': 0.15,     # Hypertension
            '42731': 0.05,    # Atrial fibrillation
            '2724': 0.04,     # Hyperlipidemia
            '5849': 0.06,     # Kidney disease
            '5712': 0.03,     # Liver disease
            '2859': 0.07,     # Anemia
            '78650': 0.04,    # Chest pain
            '486': 0.08,      # Pneumonia
            '99592': 0.05,    # Severe sepsis
            '5990': 0.07      # Urinary tract infection
        }
        
        # Generate diagnosis data
        data = []
        for _ in range(nrows):
            # Each admission has 1-8 diagnoses
            num_diagnoses = np.random.choice(range(1, 9), p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
            
            subject_id = np.random.randint(1, nrows//10)
            hadm_id = np.random.randint(1, nrows//5)
            
            # Select diagnoses
            selected_codes = np.random.choice(
                list(common_diagnoses.keys()),
                size=num_diagnoses,
                replace=False,
                p=list(common_diagnoses.values())
            )
            
            for seq_num, icd9_code in enumerate(selected_codes, 1):
                data.append({
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': hadm_id,
                    'SEQ_NUM': seq_num,
                    'ICD9_CODE': icd9_code
                })
        
        return pd.DataFrame(data[:nrows])
    
    def _create_sample_procedures(self, nrows: int) -> pd.DataFrame:
        """Create sample procedures data"""
        np.random.seed(42)
        
        # Common procedure codes
        procedures = ['9904', '8856', '3893', '8872', '9920', '8847', '3995']
        
        data = []
        for _ in range(nrows):
            data.append({
                'SUBJECT_ID': np.random.randint(1, nrows//10),
                'HADM_ID': np.random.randint(1, nrows//5),
                'SEQ_NUM': np.random.randint(1, 6),
                'ICD9_CODE': np.random.choice(procedures)
            })
        
        return pd.DataFrame(data)
    
    def _create_sample_labevents(self, nrows: int) -> pd.DataFrame:
        """Create sample lab events with realistic values"""
        np.random.seed(42)
        
        # Lab items with normal ranges
        lab_items = {
            50811: {'name': 'Hemoglobin', 'normal_range': (12, 16), 'unit': 'g/dL'},
            50824: {'name': 'Sodium', 'normal_range': (135, 145), 'unit': 'mEq/L'},
            50822: {'name': 'Potassium', 'normal_range': (3.5, 5.0), 'unit': 'mEq/L'},
            50912: {'name': 'Creatinine', 'normal_range': (0.7, 1.2), 'unit': 'mg/dL'},
            50931: {'name': 'Glucose', 'normal_range': (70, 100), 'unit': 'mg/dL'},
            51300: {'name': 'WBC Count', 'normal_range': (4, 11), 'unit': 'K/uL'}
        }
        
        data = []
        base_time = pd.Timestamp('2008-01-01')
        
        for _ in range(nrows):
            itemid = np.random.choice(list(lab_items.keys()))
            lab_info = lab_items[itemid]
            
            # Generate values (some abnormal for realism)
            if np.random.random() < 0.7:  # 70% normal values
                value = np.random.uniform(*lab_info['normal_range'])
            else:  # 30% abnormal values
                if np.random.random() < 0.5:
                    value = np.random.uniform(lab_info['normal_range'][0] * 0.5, lab_info['normal_range'][0])
                else:
                    value = np.random.uniform(lab_info['normal_range'][1], lab_info['normal_range'][1] * 1.5)
            
            data.append({
                'SUBJECT_ID': np.random.randint(1, nrows//100),
                'HADM_ID': np.random.randint(1, nrows//50),
                'ITEMID': itemid,
                'CHARTTIME': base_time + pd.Timedelta(
                    days=np.random.randint(0, 1000),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                ),
                'VALUE': round(value, 2),
                'VALUEUOM': lab_info['unit']
            })
        
        return pd.DataFrame(data)
    
    def _get_age_distribution(self) -> List[float]:
        """Realistic age distribution for hospital patients"""
        # Higher probability for older patients
        probs = []
        for age in range(18, 95):
            if age < 30:
                prob = 0.005
            elif age < 50:
                prob = 0.01
            elif age < 70:
                prob = 0.02
            else:
                prob = 0.03
            probs.append(prob)
        
        # Normalize probabilities
        total = sum(probs)
        return [p/total for p in probs]
    
    def create_readmission_features(self) -> pd.DataFrame:
        """
        Create feature dataset for readmission prediction
        This is the main function for your machine learning pipeline
        """
        logger.info("Creating readmission features dataset...")
        
        # Load required tables
        admissions = self.load_mimic_data('admissions', 2000)
        patients = self.load_mimic_data('patients', 700)
        diagnoses = self.load_mimic_data('diagnoses', 4000)
        procedures = self.load_mimic_data('procedures', 2000)
        
        # Merge basic patient and admission data
        features_df = self._create_basic_features(admissions, patients)
        
        # Add comorbidity features
        features_df = self._add_comorbidity_features(features_df, diagnoses)
        
        # Add procedure features
        features_df = self._add_procedure_features(features_df, procedures)
        
        # Add readmission target variable
        features_df = self._add_readmission_target(features_df, admissions)
        
        # Calculate LACE and HOSPITAL scores
        features_df = self._calculate_risk_scores(features_df)
        
        # Save processed dataset
        output_path = DEMO_DATA_DIR / "processed_features.csv"
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed features to {output_path}")
        
        self.processed_features = features_df
        return features_df
    
    def _create_basic_features(self, admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
        """Create basic demographic and admission features"""
        
        # Merge admissions with patients
        df = admissions.merge(patients, on='SUBJECT_ID', how='left')
        
        # Calculate age at admission
        df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
        df['DOB'] = pd.to_datetime(df['DOB'])
        df['age'] = (df['ADMITTIME'] - df['DOB']).dt.days / 365.25
        
        # Calculate length of stay
        df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
        df['length_of_stay'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.days
        df['length_of_stay'] = df['length_of_stay'].fillna(0)
        
        # Clean and encode categorical variables
        df['gender'] = df['GENDER'].map({'M': 1, 'F': 0})
        df['emergency_admission'] = (df['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)
        
        # Select relevant columns
        feature_columns = [
            'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
            'age', 'gender', 'length_of_stay', 'emergency_admission',
            'HOSPITAL_EXPIRE_FLAG', 'ETHNICITY', 'INSURANCE'
        ]
        
        return df[feature_columns].copy()
    
    def _add_comorbidity_features(self, df: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
        """Add comorbidity features based on ICD-9 codes"""
        
        # Define comorbidity mappings
        comorbidity_codes = {
            'diabetes': ['25000', '25001', '25002', '25003'],
            'hypertension': ['4019', '4010', '4011'],
            'heart_failure': ['4280', '4281', '42821'],
            'copd': ['49121', '49320'],
            'renal_failure': ['5849', '5856'],
            'liver_disease': ['5712', '5715']
        }
        
        # Count diagnoses per admission
        diagnosis_counts = diagnoses.groupby('HADM_ID').size().reset_index(name='num_diagnoses')
        df = df.merge(diagnosis_counts, on='HADM_ID', how='left')
        df['num_diagnoses'] = df['num_diagnoses'].fillna(0)
        
        # Add comorbidity flags
        for condition, codes in comorbidity_codes.items():
            condition_diagnoses = diagnoses[diagnoses['ICD9_CODE'].isin(codes)]['HADM_ID'].unique()
            df[condition] = df['HADM_ID'].isin(condition_diagnoses).astype(int)
        
        return df
    
    def _add_procedure_features(self, df: pd.DataFrame, procedures: pd.DataFrame) -> pd.DataFrame:
        """Add procedure-related features"""
        
        # Count procedures per admission
        procedure_counts = procedures.groupby('HADM_ID').size().reset_index(name='num_procedures')
        df = df.merge(procedure_counts, on='HADM_ID', how='left')
        df['num_procedures'] = df['num_procedures'].fillna(0)
        
        return df
    
    def _add_readmission_target(self, df: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
        """Add 30-day readmission target variable"""
        
        # Sort by patient and admission time
        admissions_sorted = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
        
        readmission_flags = []
        
        for _, row in df.iterrows():
            subject_id = row['SUBJECT_ID']
            current_discharge = row['DISCHTIME']
            
            # Find next admission for this patient
            patient_admissions = admissions_sorted[
                (admissions_sorted['SUBJECT_ID'] == subject_id) &
                (admissions_sorted['ADMITTIME'] > current_discharge)
            ]
            
            if len(patient_admissions) > 0 and pd.notna(current_discharge):
                next_admission = patient_admissions.iloc[0]['ADMITTIME']
                days_to_readmission = (next_admission - current_discharge).days
                
                if days_to_readmission <= 30:
                    readmission_flags.append(1)
                else:
                    readmission_flags.append(0)
            else:
                readmission_flags.append(0)
        
        df['readmission_30_day'] = readmission_flags
        return df
    
    def _calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate LACE and HOSPITAL risk scores"""
        
        # LACE Score calculation
        df['lace_length_score'] = np.minimum(df['length_of_stay'], 14)
        df['lace_acuity_score'] = df['emergency_admission'] * 3
        df['lace_comorbidity_score'] = (
            df['diabetes'] + df['hypertension'] + df['heart_failure'] + 
            df['copd'] + df['renal_failure'] + df['liver_disease']
        )
        df['lace_total'] = (
            df['lace_length_score'] + df['lace_acuity_score'] + df['lace_comorbidity_score']
        )
        
        # Simplified HOSPITAL score (without all lab values)
        df['hospital_score'] = (
            df['emergency_admission'] + 
            (df['length_of_stay'] > 5).astype(int) +
            df['num_procedures'] > 0
        )
        
        return df


def load_demo_dataset() -> pd.DataFrame:
    """
    Main function to load/create the demo dataset for your project
    """
    processor = HospitalDataProcessor()
    
    # Check if processed data already exists
    demo_file = DEMO_DATA_DIR / "processed_features.csv"
    
    if demo_file.exists():
        logger.info(f"Loading existing demo dataset from {demo_file}")
        return pd.read_csv(demo_file)
    else:
        logger.info("Creating new demo dataset...")
        return processor.create_readmission_features()


if __name__ == "__main__":
    # Test the data processor
    print(" Hospital Readmission Data Processor")
    print("=" * 50)
    
    # Create demo dataset
    df = load_demo_dataset()
    
    print(f"\nDataset Summary:")
    print(f"Total admissions: {len(df)}")
    print(f"Unique patients: {df['SUBJECT_ID'].nunique()}")
    print(f"30-day readmissions: {df['readmission_30_day'].sum()} ({df['readmission_30_day'].mean():.1%})")
    print(f"Average age: {df['age'].mean():.1f} years")
    print(f"Average length of stay: {df['length_of_stay'].mean():.1f} days")
    
    print(f"\n Common Conditions:")
    conditions = ['diabetes', 'hypertension', 'heart_failure', 'copd']
    for condition in conditions:
        if condition in df.columns:
            print(f"  {condition}: {df[condition].sum()} patients ({df[condition].mean():.1%})")
    
    print(f"\nRisk Scores:")
    if 'lace_total' in df.columns:
        print(f"  Average LACE score: {df['lace_total'].mean():.1f}")
        print(f"  High LACE risk (>10): {(df['lace_total'] > 10).sum()} patients")
    
    print("\n Demo dataset ready for machine learning!")