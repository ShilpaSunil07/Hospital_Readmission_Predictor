"""
Simple Baseline Models for Hospital Readmission Prediction
MSc Data Science Project - Shilpa Sunil (001422153)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

print(" Hospital Readmission Prediction - Baseline Models")
print("Student: Shilpa Sunil (001422153)")
print("=" * 60)

try:
    # Import ML libraries
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    print(" Machine learning libraries imported successfully")
    
    # Import our data loader
    from data_processing.data_loader import load_demo_dataset
    print("Data loader imported successfully")
    
    # Load data
    print("\n Loading dataset...")
    df = load_demo_dataset()
    print(f" Dataset loaded: {df.shape}")
    print(f" Readmission rate: {df['readmission_30_day'].mean():.1%}")
    
    # Prepare features
    print("\n Preparing features...")
    feature_columns = [
        'age', 'gender', 'length_of_stay', 'emergency_admission',
        'num_diagnoses', 'num_procedures',
        'diabetes', 'hypertension', 'heart_failure', 'copd'
    ]
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f" Available features: {available_features}")
    
    # Prepare data
    X = df[available_features].fillna(0)
    y = df['readmission_30_day']
    
    print(f" Feature matrix: {X.shape}")
    print(f" Target variable: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f" Training set: {X_train.shape}")
    print(f" Test set: {X_test.shape}")
    
    # Train Random Forest
    print("\n Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print(f" Random Forest Results:")
    print(f"   Accuracy:  {rf_accuracy:.3f}")
    print(f"   Precision: {rf_precision:.3f}")
    print(f"   Recall:    {rf_recall:.3f}")
    print(f"   F1-Score:  {rf_f1:.3f}")
    print(f"   AUC-ROC:   {rf_auc:.3f}")
    
    # Train Logistic Regression
    print("\n Training Logistic Regression...")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced'
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_pred_proba)
    
    print(f" Logistic Regression Results:")
    print(f"   Accuracy:  {lr_accuracy:.3f}")
    print(f"   Precision: {lr_precision:.3f}")
    print(f"   Recall:    {lr_recall:.3f}")
    print(f"   F1-Score:  {lr_f1:.3f}")
    print(f"   AUC-ROC:   {lr_auc:.3f}")
    
    # Feature importance (Random Forest)
    print("\n Top 5 Most Important Features (Random Forest):")
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for i, row in feature_importance.head(5).iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.3f}")
    
    # Compare with LACE score if available
    if 'lace_total' in df.columns:
        print("\n Comparison with LACE Score:")
        lace_auc = roc_auc_score(df['readmission_30_day'], df['lace_total'])
        print(f"   LACE Score AUC: {lace_auc:.3f}")
        
        best_ml_auc = max(rf_auc, lr_auc)
        improvement = ((best_ml_auc - lace_auc) / lace_auc) * 100
        print(f"   Best ML Model AUC: {best_ml_auc:.3f}")
        print(f"   Improvement over LACE: {improvement:.1f}%")
    
    # Model comparison summary
    print("\nMODEL COMPARISON SUMMARY:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)
    print(f"{'Random Forest':<20} {rf_accuracy:<10.3f} {rf_precision:<10.3f} {rf_recall:<10.3f} {rf_f1:<10.3f} {rf_auc:<10.3f}")
    print(f"{'Logistic Regression':<20} {lr_accuracy:<10.3f} {lr_precision:<10.3f} {lr_recall:<10.3f} {lr_f1:<10.3f} {lr_auc:<10.3f}")
    
    print("\n BASELINE MODELS COMPLETE!")
    print(" Phase 1: Research & Data Setup (COMPLETE)")
    print(" Phase 2: Baseline Models (COMPLETE)")
    print(" Phase 3: Advanced Models & Web App (READY)")
    
    print(f"\n PROJECT STATUS:")
    print(f"   Progress: 2.5/5 phases complete")
    print(f"   Dataset: {len(df)} admissions, {df['readmission_30_day'].mean():.1%} readmission rate")
    print(f"   Best Model: {'Random Forest' if rf_auc > lr_auc else 'Logistic Regression'}")
    print(f"   Best AUC: {max(rf_auc, lr_auc):.3f}")
    
except ImportError as e:
    print(f"Missing package: {e}")
    print("Please install required packages:")
    print("pip install pandas numpy scikit-learn")
    
except Exception as e:
    print(f" Error: {e}")
    print("Please check your data and try again")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("1. Advanced models (LSTM, Neural Networks)")
print("2. Streamlit web application")
print("3. SHAP explainability")
print("4. Clinical validation")
print("=" * 60)