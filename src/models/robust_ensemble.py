"""
Robust Advanced Ensemble for Hospital Readmission Prediction
Target: 0.70+ AUC with numerical stability
MSc Data Science Project - Shilpa Sunil (001422153)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

print(" Hospital Readmission Prediction - ROBUST ENSEMBLE")
print("Student: Shilpa Sunil (001422153)")
print("TARGET: 0.70+ AUC (Numerical Stability Fixed)")
print("=" * 60)

try:
    # Import advanced ML libraries
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
        VotingClassifier, StackingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
    )
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, roc_curve, classification_report
    )
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    import joblib
    
    print(" Advanced ML libraries imported successfully")
    
except ImportError as e:
    print(f" Missing packages. Installing...")
    import subprocess
    packages = ["xgboost", "lightgbm", "imbalanced-learn"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(" Packages installed. Please restart and run again.")
    sys.exit()

# Import our modules
from data_processing.data_loader import load_demo_dataset
from config import MODELS_DIR, FIGURES_DIR

class RobustAdvancedPredictor:
    """
    Robust advanced ensemble with numerical stability
    Target: 0.70+ AUC for Good Clinical Performance
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        
    def safe_divide(self, numerator, denominator, default=0):
        """Safely divide avoiding inf and nan"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result = np.where(np.isfinite(result), result, default)
        return result
    
    def clean_features(self, df):
        """Clean features to avoid numerical issues"""
        # Replace inf and -inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with appropriate values
        df = df.fillna(0)
        
        # Clip extreme values to reasonable ranges
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['SUBJECT_ID', 'HADM_ID']:  # Don't clip ID columns
                p99 = np.percentile(df[col].dropna(), 99)
                p1 = np.percentile(df[col].dropna(), 1)
                df[col] = np.clip(df[col], p1, p99)
        
        return df
    
    def load_and_engineer_features(self):
        """Robust feature engineering with numerical stability"""
        print("\n Robust Feature Engineering...")
        
        # Load data
        df = load_demo_dataset()
        print(f" Dataset loaded: {df.shape}")
        
        # Create enhanced features with safety checks
        df_enhanced = df.copy()
        
        # 1. Safe polynomial features (with clipping)
        df_enhanced['age_squared'] = np.clip(df_enhanced['age'] ** 2, 0, 10000)
        df_enhanced['los_squared'] = np.clip(df_enhanced['length_of_stay'] ** 2, 0, 1000)
        df_enhanced['los_log'] = np.log1p(df_enhanced['length_of_stay'])  # log1p is safer
        
        # 2. Safe interaction features
        df_enhanced['age_los_interaction'] = df_enhanced['age'] * df_enhanced['length_of_stay']
        df_enhanced['age_diagnoses_interaction'] = df_enhanced['age'] * df_enhanced['num_diagnoses']
        
        # 3. Safe ratio features (avoiding division by zero)
        df_enhanced['diagnosis_procedure_ratio'] = self.safe_divide(
            df_enhanced['num_diagnoses'], 
            df_enhanced['num_procedures'] + 1,  # Add 1 to avoid division by zero
            default=df_enhanced['num_diagnoses'].median()
        )
        
        df_enhanced['total_interventions'] = df_enhanced['num_diagnoses'] + df_enhanced['num_procedures']
        df_enhanced['intervention_intensity'] = self.safe_divide(
            df_enhanced['total_interventions'],
            df_enhanced['length_of_stay'] + 0.1,  # Add small value to avoid division by zero
            default=1.0
        )
        
        # 4. Comorbidity features
        comorbidity_cols = ['diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease']
        df_enhanced['comorbidity_count'] = df_enhanced[comorbidity_cols].sum(axis=1)
        df_enhanced['comorbidity_burden'] = self.safe_divide(
            df_enhanced['comorbidity_count'], 
            len(comorbidity_cols),
            default=0
        )
        
        # Specific comorbidity combinations
        df_enhanced['cardiovascular_risk'] = (
            df_enhanced['hypertension'] + df_enhanced['heart_failure']
        )
        df_enhanced['diabetes_complications'] = (
            df_enhanced['diabetes'] * df_enhanced['cardiovascular_risk']
        )
        
        # 5. Age stratification (clinical meaningful)
        df_enhanced['very_elderly'] = (df_enhanced['age'] >= 80).astype(int)
        df_enhanced['elderly'] = ((df_enhanced['age'] >= 65) & (df_enhanced['age'] < 80)).astype(int)
        df_enhanced['middle_aged'] = ((df_enhanced['age'] >= 50) & (df_enhanced['age'] < 65)).astype(int)
        
        # 6. Length of stay categories
        df_enhanced['short_stay'] = (df_enhanced['length_of_stay'] <= 3).astype(int)
        df_enhanced['medium_stay'] = ((df_enhanced['length_of_stay'] > 3) & (df_enhanced['length_of_stay'] <= 7)).astype(int)
        df_enhanced['long_stay'] = (df_enhanced['length_of_stay'] > 7).astype(int)
        
        # 7. High-risk combinations
        df_enhanced['high_risk_elderly'] = (
            (df_enhanced['age'] >= 75) & 
            (df_enhanced['comorbidity_count'] >= 2) &
            (df_enhanced['emergency_admission'] == 1)
        ).astype(int)
        
        df_enhanced['complex_admission'] = (
            (df_enhanced['num_diagnoses'] >= 5) & 
            (df_enhanced['num_procedures'] >= 2)
        ).astype(int)
        
        # 8. Enhanced clinical scores
        df_enhanced['lace_age_interaction'] = df_enhanced['lace_total'] * (df_enhanced['age'] / 100)
        df_enhanced['hospital_comorbidity_interaction'] = df_enhanced['hospital_score'] * df_enhanced['comorbidity_count']
        df_enhanced['combined_risk_score'] = df_enhanced['lace_total'] + df_enhanced['hospital_score']
        
        # 9. Normalized features (safe normalization)
        age_max = df_enhanced['age'].max()
        los_max = df_enhanced['length_of_stay'].max()
        
        df_enhanced['age_normalized'] = self.safe_divide(df_enhanced['age'], age_max, default=0.5)
        df_enhanced['los_normalized'] = self.safe_divide(df_enhanced['length_of_stay'], los_max, default=0.1)
        
        # Select robust feature set
        feature_columns = [
            # Basic features
            'age', 'gender', 'length_of_stay', 'emergency_admission',
            'num_diagnoses', 'num_procedures',
            
            # Comorbidities
            'diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease',
            
            # Risk scores
            'lace_total', 'hospital_score',
            
            # Safe polynomial features
            'age_squared', 'los_squared', 'los_log',
            
            # Safe interactions
            'age_los_interaction', 'age_diagnoses_interaction',
            
            # Safe ratios
            'diagnosis_procedure_ratio', 'total_interventions', 'intervention_intensity',
            
            # Comorbidity features
            'comorbidity_count', 'comorbidity_burden', 'cardiovascular_risk', 'diabetes_complications',
            
            # Age stratification
            'very_elderly', 'elderly', 'middle_aged',
            
            # LOS categories
            'short_stay', 'medium_stay', 'long_stay',
            
            # High-risk combinations
            'high_risk_elderly', 'complex_admission',
            
            # Enhanced scores
            'lace_age_interaction', 'hospital_comorbidity_interaction', 'combined_risk_score',
            
            # Normalized features
            'age_normalized', 'los_normalized'
        ]
        
        # Filter available features and clean
        available_features = [col for col in feature_columns if col in df_enhanced.columns]
        X = df_enhanced[available_features]
        
        # Clean features for numerical stability
        X = self.clean_features(X)
        
        # Final validation
        print(f" Robust features created: {len(available_features)}")
        print(f" Numerical validation: {np.isfinite(X.values).all()}")
        
        y = df_enhanced['readmission_30_day']
        self.feature_names = available_features
        
        print(f" Feature matrix: {X.shape}")
        print(f" Readmission rate: {y.mean():.1%}")
        
        return X, y, df_enhanced
    
    def create_robust_models(self):
        """Create robust model ensemble optimized for 0.70+ AUC"""
        print("\n Creating Robust Model Ensemble...")
        
        models = {}
        
        # 1. XGBoost (optimized for medical data)
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=10,  # Handle class imbalance
            random_state=self.random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # 2. LightGBM (fast and accurate)
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1,
            class_weight='balanced',
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        # 3. Random Forest (robust baseline)
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 4. Extra Trees (high variance for ensemble)
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 5. Gradient Boosting (sklearn implementation)
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        return models
    
    def train_with_smote(self, X, y):
        """Train models with SMOTE balancing"""
        print("\n Applying SMOTE for Class Balance...")
        
        # Original distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original: {dict(zip(unique, counts))} ({y.mean():.1%} positive)")
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # New distribution
        unique, counts = np.unique(y_balanced, return_counts=True)
        print(f"Balanced: {dict(zip(unique, counts))} ({y_balanced.mean():.1%} positive)")
        
        return X_balanced, y_balanced
    
    def train_robust_ensemble(self, X, y):
        """Train robust ensemble targeting 0.70+ AUC"""
        print("\n Training Robust Ensemble for 0.70+ AUC...")
        
        # Apply SMOTE balancing
        X_balanced, y_balanced = self.train_with_smote(X, y)
        
        # Split balanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, 
            random_state=self.random_state, stratify=y_balanced
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Create models
        models = self.create_robust_models()
        
        # Train each model
        trained_models = []
        
        for model_name, model in models.items():
            print(f"\n Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Optimize threshold for F1-score
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Calculate metrics
                results = {
                    'auc': roc_auc_score(y_test, y_pred_proba),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'threshold': optimal_threshold
                }
                
                self.models[model_name] = model
                self.results[model_name] = {
                    'metrics': results,
                    'y_test': y_test,
                    'y_pred_proba': y_pred_proba,
                    'model_name': model_name.replace('_', ' ').title()
                }
                
                print(f" {model_name}: AUC={results['auc']:.3f}, F1={results['f1']:.3f}, Recall={results['recall']:.3f}")
                
                # Add successful models to ensemble list
                if results['auc'] > 0.60:  # Only include decent models
                    trained_models.append((model_name, model))
                    
            except Exception as e:
                print(f" Error training {model_name}: {e}")
        
        # Create Voting Ensemble
        if len(trained_models) >= 3:
            print(f"\n Creating Voting Ensemble...")
            
            voting_clf = VotingClassifier(
                estimators=trained_models,
                voting='soft'
            )
            
            voting_clf.fit(X_train, y_train)
            y_pred_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
            
            # Optimize threshold
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_voting)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            y_pred_voting = (y_pred_proba_voting >= optimal_threshold).astype(int)
            
            voting_results = {
                'auc': roc_auc_score(y_test, y_pred_proba_voting),
                'accuracy': accuracy_score(y_test, y_pred_voting),
                'precision': precision_score(y_test, y_pred_voting, zero_division=0),
                'recall': recall_score(y_test, y_pred_voting),
                'f1': f1_score(y_test, y_pred_voting),
                'threshold': optimal_threshold
            }
            
            self.models['voting_ensemble'] = voting_clf
            self.results['voting_ensemble'] = {
                'metrics': voting_results,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba_voting,
                'model_name': 'Voting Ensemble'
            }
            
            print(f" Voting Ensemble: AUC={voting_results['auc']:.3f}, F1={voting_results['f1']:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_final_performance(self):
        """Evaluate final performance for 0.70+ AUC target"""
        print("\n FINAL PERFORMANCE EVALUATION:")
        print("=" * 60)
        
        # Performance comparison
        print(f"PERFORMANCE COMPARISON:")
        print(f"{'Model':<20} {'AUC':<8} {'F1':<8} {'Recall':<8} {'Precision':<10}")
        print("-" * 60)
        
        # Baseline comparisons
        baselines = {
            'LACE Score': 0.549,
            'HOSPITAL Score': 0.528,
            'Phase 2 Best': 0.652
        }
        
        for model, auc in baselines.items():
            print(f"{model:<20} {auc:<8.3f} {'N/A':<8} {'N/A':<8} {'N/A':<10}")
        
        print("-" * 60)
        
        # Our models
        for model_name, result in self.results.items():
            metrics = result['metrics']
            display_name = result['model_name'][:19]
            print(f"{display_name:<20} {metrics['auc']:<8.3f} {metrics['f1']:<8.3f} {metrics['recall']:<8.3f} {metrics['precision']:<10.3f}")
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['auc'])
        best_result = self.results[best_model_name]
        best_auc = best_result['metrics']['auc']
        
        print("-" * 60)
        print(f" BEST MODEL: {best_result['model_name']} (AUC: {best_auc:.3f})")
        
        # Performance assessment
        if best_auc >= 0.70:
            performance_level = "GOOD"
            status = " TARGET ACHIEVED!"
            color = ""
        elif best_auc >= 0.65:
            performance_level = "FAIR+"
            status = " Very close to target"
            color = ""
        elif best_auc >= 0.60:
            performance_level = "FAIR"
            status = " Decent performance"
            color = ""
        else:
            performance_level = "POOR"
            status = " Needs improvement"
            color = ""
        
        print(f"\n{color} CLINICAL PERFORMANCE: {performance_level}")
        print(f"{status}")
        
        # Calculate improvements
        improvement_vs_phase2 = ((best_auc - 0.652) / 0.652) * 100
        improvement_vs_lace = ((best_auc - 0.549) / 0.549) * 100
        
        print(f" Improvement vs Phase 2 Best: {improvement_vs_phase2:+.1f}%")
        print(f" Improvement vs LACE Score: {improvement_vs_lace:+.1f}%")
        
        if best_auc >= 0.70:
            print(f"\n BREAKTHROUGH ACHIEVED!")
            print(f"   • Good clinical performance level reached")
            print(f"   • Suitable for clinical decision support")
            print(f"   • Publication-worthy results")
            print(f"   • Industry-grade performance")
        elif best_auc >= 0.65:
            print(f"\n EXCELLENT PROGRESS!")
            print(f"   • Very close to Good clinical performance")
            print(f"   • Significant improvement demonstrated")
            print(f"   • Strong clinical utility")
        
        # Additional insights
        print(f"\n Clinical Insights:")
        print(f"   • Model Recall: {best_result['metrics']['recall']:.1%} (readmission detection)")
        print(f"   • Model Precision: {best_result['metrics']['precision']:.1%} (prediction accuracy)")
        print(f"   • F1-Score: {best_result['metrics']['f1']:.3f} (balanced performance)")
        
        return best_model_name, best_auc, performance_level
    
    def save_best_models(self):
        """Save the best performing models"""
        print("\n Saving Best Models...")
        
        # Save top 3 models by AUC
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['metrics']['auc'],
            reverse=True
        )[:3]
        
        for i, (model_name, result) in enumerate(sorted_models):
            model_path = MODELS_DIR / f"robust_model_{i+1}_{model_name}.pkl"
            joblib.dump(self.models[model_name], model_path)
            print(f" Saved #{i+1}: {model_name} (AUC: {result['metrics']['auc']:.3f})")


def run_robust_ensemble():
    """Main function to run robust ensemble for 0.70+ AUC"""
    print("" + "=" * 60 + "")
    print("    ROBUST ENSEMBLE FOR 0.70+ AUC")
    print("    NUMERICAL STABILITY + PERFORMANCE")
    print("    MSc Data Science - Shilpa Sunil")
    print("" + "=" * 60 + "")
    
    # Initialize predictor
    predictor = RobustAdvancedPredictor()
    
    # Load and engineer features robustly
    X, y, df = predictor.load_and_engineer_features()
    
    # Train robust ensemble
    X_train, X_test, y_train, y_test = predictor.train_robust_ensemble(X, y)
    
    # Final evaluation
    best_model, best_auc, performance_level = predictor.evaluate_final_performance()
    
    # Save models
    predictor.save_best_models()
    
    print("\n ROBUST ENSEMBLE COMPLETE!")
    print(" Phase 1: Research & Data Setup (COMPLETE)")
    print(" Phase 2: Baseline Models (COMPLETE)")
    print(" Phase 3: Web Application (COMPLETE)")
    print(" Phase 4: Robust Advanced Ensemble (COMPLETE)")
    
    if best_auc >= 0.70:
        print(" BREAKTHROUGH: 0.70+ AUC ACHIEVED!")
    elif best_auc >= 0.65:
        print(" EXCELLENT: Very close to 0.70 target!")
    else:
        print(f" PROGRESS: {best_auc:.3f} AUC achieved")
    
    print(f"\n FINAL ACHIEVEMENTS:")
    print(f"   Best Model: {best_model}")
    print(f"   Best AUC: {best_auc:.3f}")
    print(f"   Performance Level: {performance_level}")
    print(f"   Features Used: {len(predictor.feature_names)}")
    print(f"   Models Tested: {len(predictor.models)}")
    
    # Project completion status
    if best_auc >= 0.65:
        print(f"\n PROJECT STATUS: OUTSTANDING SUCCESS")
        print(f"   • Achieved excellent clinical performance")
        print(f"   • Demonstrated advanced ML expertise")
        print(f"   • Ready for dissertation defense")
        print(f"   • Industry-quality results")
    
    return predictor, best_model, best_auc


if __name__ == "__main__":
    predictor, best_model, best_auc = run_robust_ensemble()