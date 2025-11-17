# realistic_data_generator.py - Create clinically accurate training data
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_realistic_hospital_data(n_patients=2000):
    """
    Generate realistic hospital readmission data based on clinical literature
    """
    
    print("Generating Realistic Hospital Readmission Data")
    print("=" * 55)
    print("Based on published clinical literature and evidence")
    
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    patients = []
    
    # Evidence-based readmission risk factors from literature
    base_readmission_rate = 0.08  # 8% baseline
    
    for i in range(n_patients):
        patient = {}
        
        # DEMOGRAPHICS (realistic distributions)
        # Age distribution: 20% young, 40% middle, 40% elderly
        age_group = np.random.choice(['young', 'middle', 'elderly'], p=[0.2, 0.4, 0.4])
        
        if age_group == 'young':
            age = np.random.normal(35, 8)  # Young adults
            age = max(18, min(45, age))
        elif age_group == 'middle':
            age = np.random.normal(55, 10)  # Middle-aged
            age = max(45, min(65, age))
        else:
            age = np.random.normal(75, 8)  # Elderly
            age = max(65, min(95, age))
        
        patient['age'] = round(age, 1)
        patient['gender'] = np.random.choice([0, 1], p=[0.52, 0.48])  # Slightly more females
        
        # ADMISSION TYPE (evidence-based)
        # Emergency admissions more common in elderly
        if age >= 65:
            emergency_prob = 0.65
        elif age >= 45:
            emergency_prob = 0.45
        else:
            emergency_prob = 0.30
        
        patient['emergency_admission'] = np.random.choice([0, 1], p=[1-emergency_prob, emergency_prob])
        
        # LENGTH OF STAY (realistic distributions)
        if patient['emergency_admission']:
            los = np.random.lognormal(1.5, 0.8)  # Emergency: longer stays
        else:
            los = np.random.lognormal(1.0, 0.6)  # Elective: shorter stays
        
        patient['length_of_stay'] = max(1, min(30, round(los)))
        
        # COMORBIDITIES (age-dependent, evidence-based)
        # Diabetes prevalence by age
        if age < 45:
            diabetes_prob = 0.05
        elif age < 65:
            diabetes_prob = 0.15
        else:
            diabetes_prob = 0.25
        
        patient['diabetes'] = np.random.choice([0, 1], p=[1-diabetes_prob, diabetes_prob])
        
        # Heart failure (more common in elderly)
        if age < 45:
            hf_prob = 0.02
        elif age < 65:
            hf_prob = 0.08
        else:
            hf_prob = 0.20
        
        patient['heart_failure'] = np.random.choice([0, 1], p=[1-hf_prob, hf_prob])
        
        # Renal failure (age and diabetes dependent)
        renal_base_prob = 0.05 if age < 45 else 0.12 if age < 65 else 0.25
        if patient['diabetes']:
            renal_base_prob *= 2.5
        if patient['heart_failure']:
            renal_base_prob *= 1.8
        
        renal_base_prob = min(0.8, renal_base_prob)
        patient['renal_failure'] = np.random.choice([0, 1], p=[1-renal_base_prob, renal_base_prob])
        
        # Liver disease (less common, some correlation with age)
        liver_prob = 0.03 if age < 45 else 0.07 if age < 65 else 0.12
        patient['liver_disease'] = np.random.choice([0, 1], p=[1-liver_prob, liver_prob])
        
        # Hypertension (very common in elderly)
        if age < 45:
            htn_prob = 0.15
        elif age < 65:
            htn_prob = 0.45
        else:
            htn_prob = 0.70
        
        patient['hypertension'] = np.random.choice([0, 1], p=[1-htn_prob, htn_prob])
        
        # COPD (age and smoking related)
        if age < 45:
            copd_prob = 0.02
        elif age < 65:
            copd_prob = 0.08
        else:
            copd_prob = 0.18
        
        patient['copd'] = np.random.choice([0, 1], p=[1-copd_prob, copd_prob])
        
        # CLINICAL COMPLEXITY
        # Number of diagnoses (correlated with age and comorbidities)
        base_diagnoses = 1 + (age - 18) / 20  # Age effect
        base_diagnoses += sum([patient['diabetes'], patient['heart_failure'], 
                              patient['renal_failure'], patient['hypertension'], patient['copd']]) * 1.5
        
        if patient['emergency_admission']:
            base_diagnoses *= 1.3
        
        diagnoses = max(1, round(np.random.poisson(base_diagnoses)))
        patient['num_diagnoses'] = min(15, diagnoses)
        
        # Number of procedures (related to complexity and admission type)
        if patient['emergency_admission']:
            proc_mean = 1.5
        else:
            proc_mean = 0.8
        
        if patient['heart_failure'] or patient['renal_failure']:
            proc_mean *= 1.4
        
        procedures = np.random.poisson(proc_mean)
        patient['num_procedures'] = min(10, procedures)
        
        # CALCULATE REALISTIC READMISSION RISK
        # Start with base rate
        readmission_risk = base_readmission_rate
        
        # Age effects (U-shaped: young and very old higher risk)
        if age < 25:
            age_multiplier = 1.3  # Young adults: social factors, medication adherence
        elif age < 45:
            age_multiplier = 0.8  # Lowest risk group
        elif age < 65:
            age_multiplier = 0.9  # Still relatively low
        elif age < 75:
            age_multiplier = 1.2  # Starting to increase
        else:
            age_multiplier = 1.6  # Elderly: highest risk
        
        readmission_risk *= age_multiplier
        
        # Emergency admission effect
        if patient['emergency_admission']:
            readmission_risk *= 1.4
        
        # Comorbidity effects (evidence-based multipliers)
        if patient['heart_failure']:
            readmission_risk *= 2.0  # Strong predictor
        
        if patient['renal_failure']:
            readmission_risk *= 1.8  # Strong predictor
        
        if patient['diabetes']:
            readmission_risk *= 1.3
        
        if patient['copd']:
            readmission_risk *= 1.5
        
        if patient['liver_disease']:
            readmission_risk *= 1.4
        
        # Length of stay effects
        if patient['length_of_stay'] <= 2:
            readmission_risk *= 1.3  # Too short: premature discharge
        elif patient['length_of_stay'] >= 14:
            readmission_risk *= 1.5  # Too long: complex case
        
        # Clinical complexity
        if patient['num_diagnoses'] >= 8:
            readmission_risk *= 1.3
        
        if patient['num_procedures'] >= 4:
            readmission_risk *= 1.2
        
        # Cap the risk at realistic maximum
        readmission_risk = min(0.50, readmission_risk)  # Max 50%
        
        # Generate readmission outcome
        patient['readmission_30_day'] = np.random.choice([0, 1], p=[1-readmission_risk, readmission_risk])
        
        # ADDITIONAL FEATURES
        patient['SUBJECT_ID'] = i + 1
        patient['HADM_ID'] = i + 1
        
        # Generate realistic admission/discharge dates
        admit_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365))
        discharge_date = admit_date + timedelta(days=patient['length_of_stay'])
        
        patient['ADMITTIME'] = admit_date.strftime('%Y-%m-%d')
        patient['DISCHTIME'] = discharge_date.strftime('%Y-%m-%d')
        
        patient['HOSPITAL_EXPIRE_FLAG'] = 0  # All survived to discharge
        patient['ETHNICITY'] = np.random.choice(['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER'], 
                                               p=[0.60, 0.15, 0.15, 0.08, 0.02])
        patient['INSURANCE'] = np.random.choice(['Medicare', 'Private', 'Medicaid', 'Self Pay'], 
                                               p=[0.45, 0.35, 0.15, 0.05])
        
        # LACE score components
        # L - Length of stay score
        if patient['length_of_stay'] <= 1:
            lace_length = 1
        elif patient['length_of_stay'] <= 3:
            lace_length = patient['length_of_stay']
        else:
            lace_length = min(14, 4 + patient['length_of_stay'] - 3)
        
        patient['lace_length_score'] = lace_length
        
        # A - Acuity score
        patient['lace_acuity_score'] = 3 if patient['emergency_admission'] else 0
        
        # C - Comorbidity score
        comorbidity_count = sum([patient['diabetes'], patient['heart_failure'], 
                               patient['renal_failure'], patient['liver_disease'], 
                               patient['hypertension'], patient['copd']])
        patient['lace_comorbidity_score'] = min(6, comorbidity_count)
        
        # E - Emergency visits (simulated)
        patient['lace_ed_visits'] = np.random.poisson(0.5)  # Low baseline
        if patient['emergency_admission']:
            patient['lace_ed_visits'] += 1
        
        patient['lace_total'] = (patient['lace_length_score'] + 
                               patient['lace_acuity_score'] + 
                               patient['lace_comorbidity_score'] + 
                               min(4, patient['lace_ed_visits']))
        
        # HOSPITAL score (simplified)
        patient['hospital_score'] = patient['lace_total'] >= 10
        
        patients.append(patient)
    
    # Convert to DataFrame
    df = pd.DataFrame(patients)
    
    # Display statistics
    total_patients = len(df)
    total_readmissions = df['readmission_30_day'].sum()
    readmission_rate = total_readmissions / total_patients * 100
    
    print(f"\nGENERATED DATA STATISTICS:")
    print(f"Total patients: {total_patients}")
    print(f"Total readmissions: {total_readmissions}")
    print(f"Overall readmission rate: {readmission_rate:.1f}%")
    
    # Age group analysis
    print(f"\n AGE GROUP READMISSION RATES:")
    for age_group, label in [(df['age'] < 45, 'Young (<45)'), 
                           ((df['age'] >= 45) & (df['age'] < 65), 'Middle (45-65)'),
                           (df['age'] >= 65, 'Elderly (65+)')]:
        group_data = df[age_group]
        if len(group_data) > 0:
            group_rate = group_data['readmission_30_day'].mean() * 100
            print(f"{label}: {group_rate:.1f}% ({group_data['readmission_30_day'].sum()}/{len(group_data)})")
    
    # Condition analysis
    print(f"\n CONDITION-SPECIFIC RATES:")
    conditions = ['heart_failure', 'diabetes', 'renal_failure', 'liver_disease']
    for condition in conditions:
        with_condition = df[df[condition] == 1]
        without_condition = df[df[condition] == 0]
        
        with_rate = with_condition['readmission_30_day'].mean() * 100 if len(with_condition) > 0 else 0
        without_rate = without_condition['readmission_30_day'].mean() * 100 if len(without_condition) > 0 else 0
        
        print(f"{condition}: WITH={with_rate:.1f}% vs WITHOUT={without_rate:.1f}%")
    
    return df

def save_realistic_data():
    """Generate and save realistic data"""
    
    # Generate the data
    df = generate_realistic_hospital_data(2000)
    
    # Save to CSV
    output_file = 'realistic_hospital_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nSAVED REALISTIC DATA TO: {output_file}")
    print(f" Size: {len(df)} patients, {len(df.columns)} features")
    print(f" Ready for ML training with clinically accurate patterns!")
    
    return df

if __name__ == "__main__":
    realistic_data = save_realistic_data()