"""
 Hospital Readmission Prediction - FIXED AUC 0.70+ OPTIMIZATION
Student: Shilpa Sunil (001422153)
TARGET: ACHIEVE GOOD CLINICAL PERFORMANCE (AUC ≥ 0.70)
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Standard ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Try to import imbalanced-learn (install if needed)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_AVAILABLE = True
    print(" Imbalanced-learn available")
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("Imbalanced-learn not available, using class weights")

# Data loader fallback
class DataLoader:
    """Fallback data loader with proper preprocessing"""
    def load_data(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, 'data', 'demo', 'processed_features.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f" Data loaded from {data_path}")
            return df
        else:
            print(f" Data file not found: {data_path}")
            raise FileNotFoundError(f"Cannot find data file: {data_path}")

class FixedAUCOptimizer:
    """Fixed AUC optimizer with proper data preprocessing"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.best_model = None
        self.best_auc = 0.0
        self.best_features = None
        self.optimization_results = {}
        
    def clean_and_preprocess_data(self, df):
        """Clean data and handle non-numeric columns properly"""
        print(" Cleaning and preprocessing data...")
        
        df_clean = df.copy()
        
        # 1. Handle date columns - convert to numeric features
        date_columns = []
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Check if it looks like a date
                sample_values = df_clean[col].dropna().head()
                if len(sample_values) > 0:
                    sample_val = str(sample_values.iloc[0])
                    if '-' in sample_val and len(sample_val) == 10:  # Looks like YYYY-MM-DD
                        date_columns.append(col)
        
        # Convert dates to useful features
        for col in date_columns:
            print(f"   Converting date column: {col}")
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
            # Extract useful features from dates
            df_clean[f'{col}_year'] = df_clean[col].dt.year
            df_clean[f'{col}_month'] = df_clean[col].dt.month
            df_clean[f'{col}_dayofweek'] = df_clean[col].dt.dayofweek
            
            # Calculate days since epoch (for relative timing)
            epoch = pd.Timestamp('2000-01-01')
            df_clean[f'{col}_days_since_2000'] = (df_clean[col] - epoch).dt.days
            
            # Drop original date column
            df_clean = df_clean.drop(columns=[col])
        
        # 2. Handle other object columns
        object_columns = df_clean.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in object_columns:
            if col not in ['readmission_30_day']:  # Don't encode target
                print(f"   Encoding categorical column: {col}")
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                label_encoders[col] = le
        
        # 3. Fill missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 4. Ensure all features are numeric
        feature_cols = [col for col in df_clean.columns if col != 'readmission_30_day']
        for col in feature_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(0)
        
        print(f" Data cleaned: {len(date_columns)} date columns converted, {len(object_columns)} categorical columns encoded")
        return df_clean, label_encoders
        
    def advanced_feature_engineering(self, df):
        """Create advanced features from cleaned data"""
        print(" Creating advanced features...")
        
        df_enhanced = df.copy()
        
        # Basic features that should exist
        basic_features = ['age', 'length_of_stay', 'num_diagnoses', 'num_procedures']
        available_features = [f for f in basic_features if f in df.columns]
        
        if 'age' in df.columns:
            # Age-based features
            df_enhanced['age_squared'] = df['age'] ** 2
            df_enhanced['age_log'] = np.log1p(df['age'])
            df_enhanced['age_over_65'] = (df['age'] > 65).astype(int)
            df_enhanced['age_over_75'] = (df['age'] > 75).astype(int)
        
        if 'length_of_stay' in df.columns:
            # Length of stay features
            df_enhanced['los_squared'] = df['length_of_stay'] ** 2
            df_enhanced['los_log'] = np.log1p(df['length_of_stay'])
            df_enhanced['los_long'] = (df['length_of_stay'] > 7).astype(int)
            df_enhanced['los_very_long'] = (df['length_of_stay'] > 14).astype(int)
        
        # Comorbidity features (if available)
        comorbidity_cols = ['diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease']
        available_comorbidities = [c for c in comorbidity_cols if c in df.columns]
        
        if available_comorbidities:
            df_enhanced['comorbidity_count'] = df[available_comorbidities].sum(axis=1)
            df_enhanced['has_multiple_comorbidities'] = (df_enhanced['comorbidity_count'] >= 2).astype(int)
            df_enhanced['has_serious_comorbidities'] = 0
            
            # Weight serious conditions more
            if 'heart_failure' in df.columns:
                df_enhanced['has_serious_comorbidities'] += df['heart_failure'] * 2
            if 'renal_failure' in df.columns:
                df_enhanced['has_serious_comorbidities'] += df['renal_failure'] * 2
        
        # Complexity features
        if 'num_diagnoses' in df.columns and 'num_procedures' in df.columns:
            df_enhanced['complexity_score'] = df['num_diagnoses'] * 0.3 + df['num_procedures'] * 0.7
            df_enhanced['high_complexity'] = (df_enhanced['complexity_score'] > df_enhanced['complexity_score'].quantile(0.75)).astype(int)
        
        # Emergency admission feature
        if 'emergency_admission' in df.columns:
            df_enhanced['emergency_high_risk'] = df['emergency_admission']
            if 'age' in df.columns:
                df_enhanced['emergency_elderly'] = (df['emergency_admission'] == 1) & (df['age'] > 70)
        
        # Gender interactions (if available)
        if 'gender' in df.columns and 'age' in df.columns:
            df_enhanced['male_elderly'] = (df['gender'] == 1) & (df['age'] > 70)
            df_enhanced['female_elderly'] = (df['gender'] == 0) & (df['age'] > 70)
        
        # Clinical score enhancements (if available)
        if 'lace_total' in df.columns:
            df_enhanced['lace_high_risk'] = (df['lace_total'] > 10).astype(int)
            df_enhanced['lace_squared'] = df['lace_total'] ** 2
        
        if 'hospital_score' in df.columns:
            df_enhanced['hospital_high_risk'] = (df['hospital_score'] > 5).astype(int)
        
        print(f" Enhanced features created: {df_enhanced.shape[1] - df.shape[1]} new features")
        return df_enhanced
    
    def load_and_engineer_features(self):
        """Load data and create advanced feature engineering with proper cleaning"""
        print(" Loading and engineering features...")
        
        # Load base dataset
        df_raw = self.data_loader.load_data()
        
        # Clean and preprocess
        df_clean, label_encoders = self.clean_and_preprocess_data(df_raw)
        
        # Advanced feature engineering
        df_enhanced = self.advanced_feature_engineering(df_clean)
        
        # Prepare features and target
        feature_cols = [col for col in df_enhanced.columns if col != 'readmission_30_day']
        X = df_enhanced[feature_cols]
        y = df_enhanced['readmission_30_day']
        
        # Final check - ensure all data is numeric
        X = X.select_dtypes(include=[np.number])
        
        # Remove any infinite or very large values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Remove constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            print(f"   Removing {len(constant_cols)} constant columns")
            X = X.drop(columns=constant_cols)
        
        print(f" Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f" Readmission rate: {y.mean():.1%}")
        print(f" Feature types: {X.dtypes.value_counts().to_dict()}")
        
        return X, y, df_enhanced
    
    def optimize_sampling_strategy(self, X, y):
        """Test different sampling strategies for class imbalance"""
        print(" Testing sampling strategies...")
        
        strategies = {'none': (X, y)}
        
        # Only use imbalanced-learn if available and we have enough minority samples
        minority_count = sum(y)
        if IMBALANCED_AVAILABLE and minority_count >= 6:  # Need at least 6 for k=5 neighbors
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
                X_smote, y_smote = smote.fit_resample(X, y)
                strategies['smote'] = (X_smote, y_smote)
                print(f"   SMOTE: {sum(y_smote)} positive samples")
            except Exception as e:
                print(f"   SMOTE failed: {e}")
        
        # Test each strategy
        best_strategy = 'none'
        best_auc = 0.0
        
        for name, (X_strat, y_strat) in strategies.items():
            try:
                # Use simpler model for testing
                lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                cv_scores = cross_val_score(lr, X_strat, y_strat, cv=3, scoring='roc_auc')
                avg_auc = cv_scores.mean()
                print(f"   {name}: AUC {avg_auc:.3f}")
                
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_strategy = name
            except Exception as e:
                print(f"   {name}: Failed - {e}")
        
        print(f"Best sampling strategy: {best_strategy} (AUC: {best_auc:.3f})")
        return strategies[best_strategy]
    
    def optimize_feature_selection(self, X, y):
        """Optimize feature selection for best performance"""
        print(" Optimizing feature selection...")
        
        # Test different numbers of features
        max_features = min(25, X.shape[1])  # Cap at 25 features
        feature_counts = [10, 15, 20, max_features]
        if X.shape[1] not in feature_counts:
            feature_counts.append(X.shape[1])
        
        best_features = list(X.columns)
        best_auc = 0.0
        
        for k in feature_counts:
            try:
                if k >= X.shape[1]:
                    selected_features = list(X.columns)
                else:
                    # Use SelectKBest with f_classif
                    selector = SelectKBest(score_func=f_classif, k=k)
                    X_selected = selector.fit_transform(X, y)
                    selected_features = X.columns[selector.get_support()].tolist()
                
                # Quick test
                X_test = X[selected_features]
                lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                cv_scores = cross_val_score(lr, X_test, y, cv=3, scoring='roc_auc')
                avg_auc = cv_scores.mean()
                
                print(f"   {k} features: AUC {avg_auc:.3f}")
                
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_features = selected_features
                    
            except Exception as e:
                print(f"   {k} features: Failed - {e}")
        
        print(f" Best feature count: {len(best_features)} (AUC: {best_auc:.3f})")
        return best_features
    
    def create_optimized_models(self):
        """Create optimized model configurations"""
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=2000,
                class_weight='balanced'
            ),
            
            'random_forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
        }
        
        return models
    
    def hyperparameter_optimization(self, X, y, model_name, base_model):
        """Simplified hyperparameter optimization"""
        print(f" Optimizing {model_name}...")
        
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            
            'gradient_boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6]
            }
        }
        
        if model_name not in param_grids:
            return base_model
        
        try:
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Best CV AUC: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"   Optimization failed: {e}")
            return base_model
    
    def run_optimization(self):
        """Run complete optimization process"""
        print(" HOSPITAL READMISSION - FIXED AUC 0.70+ OPTIMIZATION")
        print("=" * 60)
        print("Student: Shilpa Sunil (001422153)")
        print("TARGET: ACHIEVE GOOD CLINICAL PERFORMANCE (AUC ≥ 0.70)")
        print("=" * 60)
        
        # 1. Load and engineer features
        X, y, df = self.load_and_engineer_features()
        
        # 2. Optimize sampling strategy
        X_sampled, y_sampled = self.optimize_sampling_strategy(X, y)
        
        # 3. Optimize feature selection
        best_features = self.optimize_feature_selection(X_sampled, y_sampled)
        X_final = X_sampled[best_features]
        
        # 4. Create and optimize models
        models = self.create_optimized_models()
        
        optimized_models = {}
        for name, model in models.items():
            optimized_models[name] = self.hyperparameter_optimization(X_final, y_sampled, name, model)
        
        # 5. Final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test all models
        results = {}
        
        print("\n FINAL MODEL COMPARISON:")
        print("-" * 50)
        
        for name, model in optimized_models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                results[name] = auc
                print(f"{name:25} AUC: {auc:.3f}")
            except Exception as e:
                print(f"{name:25} Failed: {e}")
                results[name] = 0.0
        
        # Find best model
        if results:
            best_model_name = max(results, key=results.get)
            best_auc = results[best_model_name]
            
            print(f"\n BEST MODEL: {best_model_name}")
            print(f" BEST AUC: {best_auc:.3f}")
            
            if best_auc >= 0.70:
                print(" SUCCESS! Achieved AUC ≥ 0.70 (Good Clinical Performance)")
            elif best_auc >= 0.65:
                print(" GOOD! AUC ≥ 0.65 (Fair-to-Good Clinical Performance)")
            else:
                print(f" Current: {best_auc:.3f} | Target: 0.70 | Gap: {0.70 - best_auc:.3f}")
            
            # Store results
            self.best_auc = best_auc
            self.best_model = optimized_models[best_model_name]
            self.best_features = best_features
            self.optimization_results = results
            
            return self.best_model, self.best_auc, results
        else:
            print(" All models failed")
            return None, 0.0, {}

if __name__ == "__main__":
    optimizer = FixedAUCOptimizer()
    best_model, best_auc, results = optimizer.run_optimization()
    
    print("\n" + "="*60)
    print(" OPTIMIZATION COMPLETE!")
    if best_auc > 0:
        print(f" ACHIEVED AUC: {best_auc:.3f}")
        if best_auc >= 0.70:
            print(" TARGET REACHED: Good Clinical Performance!")
        elif best_auc >= 0.652:
            print(" IMPROVED: Better than baseline (0.652)!")
        else:
            print(f" PROGRESS: Working toward 0.70+ target")
    else:
        print(" Optimization failed - check data and preprocessing")
    print("="*60)