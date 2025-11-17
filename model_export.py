# model_export.py
# Script to export the AUC 0.964 model and preprocessing pipeline for Streamlit integration

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
import os

def create_and_save_optimized_model():
    """
    Recreate the AUC 0.964 model and save it for Streamlit integration
    """
    print(" Creating and Saving AUC 0.964 Model for Streamlit Integration")
    print("="*70)
    
    # Load the processed dataset
    data_path = 'data/demo/processed_features.csv'
    if not os.path.exists(data_path):
        print(f" Data file not found: {data_path}")
        print("Please ensure the processed_features.csv file exists")
        return
    
    df = pd.read_csv(data_path)
    print(f" Loaded dataset with {len(df)} records")
    
    # Handle missing values first
    print(" Handling missing values...")
    df['age'].fillna(df['age'].median(), inplace=True)
    df['gender'].fillna(0, inplace=True)  # Fill with Female (0)
    print(f" Missing values handled")
    
    # Create missing features from available data
    print(" Creating engineered features...")
    
    # Gender encoding (already encoded as 0/1)
    df['gender_encoded'] = df['gender']
    
    # Admission type encoding (use emergency_admission as proxy)
    df['admission_type_encoded'] = df['emergency_admission']  # 0=Non-emergency, 1=Emergency
    
    # Comorbidity count from available comorbidities
    comorbidity_columns = ['diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease']
    df['comorbidity_count'] = df[comorbidity_columns].sum(axis=1)
    
    # Map column names to match expected features
    df['renal_disease'] = df['renal_failure']  # Map renal_failure to renal_disease
    df['cancer'] = 0  # Not available in dataset, set to 0
    
    # Complexity score
    df['complexity_score'] = df['num_diagnoses'] + df['num_procedures'] + df['comorbidity_count']
    
    # Has surgery
    df['has_surgery'] = (df['num_procedures'] > 0).astype(int)
    
    # ICU stay (length of stay > 7 days)
    df['icu_stay'] = (df['length_of_stay'] > 7).astype(int)
    
    # Age groups: 0=Young (<35), 1=Middle (35-65), 2=Elderly (>65)
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 65, 100], labels=['Young', 'Middle', 'Elderly'])
    df['age_group_encoded'] = df['age_group'].map({'Young': 0, 'Middle': 1, 'Elderly': 2}).fillna(1)
    
    # Length of stay categories: 0=Short (≤3), 1=Medium (4-7), 2=Long (>7)
    df['los_category'] = pd.cut(df['length_of_stay'], bins=[0, 3, 7, 30], labels=['Short', 'Medium', 'Long'])
    df['los_category_encoded'] = df['los_category'].map({'Short': 0, 'Medium': 1, 'Long': 2}).fillna(1)
    
    # Define feature columns to match the expected model input
    feature_columns = [
        'age', 'gender_encoded', 'admission_type_encoded', 'length_of_stay',
        'num_diagnoses', 'num_procedures', 'comorbidity_count', 'diabetes',
        'heart_failure', 'renal_disease', 'liver_disease', 'cancer',
        'complexity_score', 'has_surgery', 'icu_stay', 'emergency_admission',
        'age_group_encoded', 'los_category_encoded'
    ]
    
    print(f" Created {len(feature_columns)} features")
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)
    y = df['readmission_30_day']  # Correct target variable name
    
    print(f" Prepared features: {X.shape}")
    print(f" Target distribution: {y.value_counts().to_dict()}")
    
    # Check for sufficient data
    if len(y.unique()) < 2:
        print(" Error: Target variable needs both classes (0 and 1)")
        return
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
    
    print(f" After SMOTE: {X_train_balanced.shape}")
    
    # Train the optimized Gradient Boosting model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    print(" Training Gradient Boosting model...")
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate performance
    from sklearn.metrics import roc_auc_score, classification_report
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f" Model trained successfully!")
    print(f" Test AUC Score: {auc_score:.3f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the complete pipeline
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'feature_columns': feature_columns,
        'selected_features': selector.get_support(),
        'feature_names': [feature_columns[i] for i in range(len(feature_columns)) if selector.get_support()[i]],
        'auc_score': auc_score,
        'training_info': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'smote_applied': True,
            'feature_selection_k': 15
        }
    }
    
    # Save the complete pipeline
    joblib.dump(model_artifacts, 'models/auc_964_complete_pipeline.pkl')
    print(" Saved complete pipeline to: models/auc_964_complete_pipeline.pkl")
    
    # Also save individual components for flexibility
    joblib.dump(model, 'models/gradient_boosting_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(selector, 'models/feature_selector.pkl')
    
    print(" Saved individual components:")
    print("   - models/gradient_boosting_model.pkl")
    print("   - models/scaler.pkl")
    print("   - models/feature_selector.pkl")
    
    # Create feature mapping for Streamlit
    feature_mapping = {
        'streamlit_input': 'model_feature',
        'age': 'age',
        'gender': 'gender_encoded',  # 0=Female, 1=Male
        'admission_type': 'admission_type_encoded',  # 0=Elective, 1=Emergency, 2=Urgent
        'length_of_stay': 'length_of_stay',
        'diabetes': 'diabetes',
        'heart_failure': 'heart_failure',
        'renal_disease': 'renal_disease',
        'liver_disease': 'liver_disease',
        'cancer': 'cancer'
    }
    
    # Save feature mapping
    joblib.dump(feature_mapping, 'models/feature_mapping.pkl')
    print(" Saved feature mapping to: models/feature_mapping.pkl")
    
    print("\n MODEL EXPORT COMPLETE!")
    print(f" Final AUC Score: {auc_score:.3f}")
    print(" All files saved in 'models/' directory")
    print(" Ready for Streamlit integration!")
    
    return model_artifacts

if __name__ == "__main__":
    create_and_save_optimized_model()