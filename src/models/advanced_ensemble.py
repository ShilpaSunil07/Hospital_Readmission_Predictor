"""
Advanced Ensemble Models for Hospital Readmission Prediction
Target: Achieve 0.70+ AUC for Good Clinical Performance
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

print("Hospital Readmission Prediction - ADVANCED ENSEMBLE")
print("Student: Shilpa Sunil (001422153)")
print("TARGET: 0.70+ AUC (Good Clinical Performance)")
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
    from sklearn.svm import SVC
    from sklearn.model_selection import (
        train_test_split, cross_val_score, StratifiedKFold, 
        GridSearchCV, RandomizedSearchCV
    )
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, roc_curve, classification_report, confusion_matrix
    )
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    import matplotlib.pyplot as plt
    import seaborn as sns
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

class AdvancedEnsemblePredictor:
    """
    Advanced ensemble methods targeting 0.70+ AUC performance
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        self.best_features = []
        
    def load_and_engineer_features(self):
        """Advanced feature engineering for maximum predictive power"""
        print("\n Advanced Feature Engineering...")
        
        # Load data
        df = load_demo_dataset()
        print(f" Dataset loaded: {df.shape}")
        
        # Create extensive feature set
        df_enhanced = df.copy()
        
        # 1. Polynomial and interaction features
        df_enhanced['age_squared'] = df_enhanced['age'] ** 2
        df_enhanced['age_cubed'] = df_enhanced['age'] ** 3
        df_enhanced['los_squared'] = df_enhanced['length_of_stay'] ** 2
        df_enhanced['los_log'] = np.log1p(df_enhanced['length_of_stay'])
        
        # 2. Age-based interactions
        df_enhanced['age_los_interaction'] = df_enhanced['age'] * df_enhanced['length_of_stay']
        df_enhanced['age_diagnoses_interaction'] = df_enhanced['age'] * df_enhanced['num_diagnoses']
        df_enhanced['age_emergency_interaction'] = df_enhanced['age'] * df_enhanced['emergency_admission']
        
        # 3. Clinical complexity features
        df_enhanced['diagnosis_procedure_ratio'] = (
            df_enhanced['num_diagnoses'] / (df_enhanced['num_procedures'] + 1)
        )
        df_enhanced['total_interventions'] = df_enhanced['num_diagnoses'] + df_enhanced['num_procedures']
        df_enhanced['intervention_intensity'] = df_enhanced['total_interventions'] / df_enhanced['length_of_stay']
        
        # 4. Comorbidity burden and patterns
        comorbidity_cols = ['diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease']
        df_enhanced['comorbidity_count'] = df_enhanced[comorbidity_cols].sum(axis=1)
        df_enhanced['comorbidity_burden'] = df_enhanced['comorbidity_count'] / len(comorbidity_cols)
        
        # Specific comorbidity combinations
        df_enhanced['cardiovascular_risk'] = (
            df_enhanced['hypertension'] + df_enhanced['heart_failure']
        )
        df_enhanced['diabetes_complications'] = (
            df_enhanced['diabetes'] * (df_enhanced['renal_failure'] + df_enhanced['cardiovascular_risk'])
        )
        df_enhanced['organ_failure_count'] = df_enhanced['renal_failure'] + df_enhanced['liver_disease']
        
        # 5. Age stratification (clinical meaningful cutoffs)
        df_enhanced['very_elderly'] = (df_enhanced['age'] >= 80).astype(int)
        df_enhanced['elderly'] = ((df_enhanced['age'] >= 65) & (df_enhanced['age'] < 80)).astype(int)
        df_enhanced['middle_aged'] = ((df_enhanced['age'] >= 50) & (df_enhanced['age'] < 65)).astype(int)
        
        # 6. Length of stay categories (clinical meaningful)
        df_enhanced['very_short_stay'] = (df_enhanced['length_of_stay'] <= 1).astype(int)
        df_enhanced['short_stay'] = ((df_enhanced['length_of_stay'] > 1) & (df_enhanced['length_of_stay'] <= 3)).astype(int)
        df_enhanced['medium_stay'] = ((df_enhanced['length_of_stay'] > 3) & (df_enhanced['length_of_stay'] <= 7)).astype(int)
        df_enhanced['long_stay'] = ((df_enhanced['length_of_stay'] > 7) & (df_enhanced['length_of_stay'] <= 14)).astype(int)
        df_enhanced['very_long_stay'] = (df_enhanced['length_of_stay'] > 14).astype(int)
        
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
        
        df_enhanced['critical_case'] = (
            (df_enhanced['emergency_admission'] == 1) &
            (df_enhanced['length_of_stay'] > 10) &
            (df_enhanced['comorbidity_count'] >= 3)
        ).astype(int)
        
        # 8. LACE and HOSPITAL score enhancements
        df_enhanced['lace_age_interaction'] = df_enhanced['lace_total'] * (df_enhanced['age'] / 100)
        df_enhanced['hospital_comorbidity_interaction'] = df_enhanced['hospital_score'] * df_enhanced['comorbidity_count']
        df_enhanced['combined_risk_score'] = df_enhanced['lace_total'] + df_enhanced['hospital_score']
        
        # 9. Normalized features
        df_enhanced['age_normalized'] = df_enhanced['age'] / 100
        df_enhanced['los_normalized'] = df_enhanced['length_of_stay'] / df_enhanced['length_of_stay'].max()
        
        # 10. Binned features for tree models
        df_enhanced['age_bin'] = pd.cut(df_enhanced['age'], bins=5, labels=False)
        df_enhanced['los_bin'] = pd.cut(df_enhanced['length_of_stay'], bins=5, labels=False)
        df_enhanced['diagnoses_bin'] = pd.cut(df_enhanced['num_diagnoses'], bins=3, labels=False)
        
        # Select all engineered features
        feature_columns = [
            # Basic features
            'age', 'gender', 'length_of_stay', 'emergency_admission',
            'num_diagnoses', 'num_procedures',
            
            # Comorbidities
            'diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease',
            
            # Risk scores
            'lace_total', 'hospital_score',
            
            # Polynomial features
            'age_squared', 'age_cubed', 'los_squared', 'los_log',
            
            # Interactions
            'age_los_interaction', 'age_diagnoses_interaction', 'age_emergency_interaction',
            
            # Clinical complexity
            'diagnosis_procedure_ratio', 'total_interventions', 'intervention_intensity',
            
            # Comorbidity features
            'comorbidity_count', 'comorbidity_burden', 'cardiovascular_risk', 
            'diabetes_complications', 'organ_failure_count',
            
            # Age stratification
            'very_elderly', 'elderly', 'middle_aged',
            
            # LOS categories
            'very_short_stay', 'short_stay', 'medium_stay', 'long_stay', 'very_long_stay',
            
            # High-risk combinations
            'high_risk_elderly', 'complex_admission', 'critical_case',
            
            # Enhanced scores
            'lace_age_interaction', 'hospital_comorbidity_interaction', 'combined_risk_score',
            
            # Normalized features
            'age_normalized', 'los_normalized',
            
            # Binned features
            'age_bin', 'los_bin', 'diagnoses_bin'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df_enhanced.columns]
        print(f"Total engineered features: {len(available_features)}")
        
        # Prepare data
        X = df_enhanced[available_features].fillna(0)
        y = df_enhanced['readmission_30_day']
        
        self.feature_names = available_features
        
        print(f" Feature matrix: {X.shape}")
        print(f" Readmission rate: {y.mean():.1%}")
        
        return X, y, df_enhanced
    
    def feature_selection_optimization(self, X, y):
        """Optimize feature selection for maximum performance"""
        print("\n Optimizing Feature Selection...")
        
        # Split for feature selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Univariate feature selection
        selector_univariate = SelectKBest(score_func=f_classif, k=25)
        X_train_univariate = selector_univariate.fit_transform(X_train_scaled, y_train)
        
        # 2. Recursive Feature Elimination with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rfe_selector = RFE(rf_selector, n_features_to_select=20, step=5)
        X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_train)
        
        # 3. Feature importance from XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = xgb_model.feature_importances_
        importance_indices = np.argsort(feature_importance)[-25:]  # Top 25 features
        
        # Test different feature sets
        feature_sets = {
            'univariate': selector_univariate.get_support(),
            'rfe': rfe_selector.support_,
            'xgb_importance': np.isin(range(len(self.feature_names)), importance_indices)
        }
        
        best_features = None
        best_score = 0
        
        for method, feature_mask in feature_sets.items():
            X_train_subset = X_train_scaled[:, feature_mask]
            X_test_subset = X_test_scaled[:, feature_mask]
            
            # Quick evaluation with logistic regression
            lr_eval = LogisticRegression(random_state=self.random_state, max_iter=1000)
            lr_eval.fit(X_train_subset, y_train)
            score = roc_auc_score(y_test, lr_eval.predict_proba(X_test_subset)[:, 1])
            
            print(f"   {method}: {np.sum(feature_mask)} features, AUC: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_features = feature_mask
        
        # Store best features
        self.best_features = [feat for i, feat in enumerate(self.feature_names) if best_features[i]]
        print(f" Best feature set: {len(self.best_features)} features, AUC: {best_score:.3f}")
        
        return X[:, best_features], self.best_features
    
    def handle_class_imbalance(self, X, y):
        """Apply advanced class imbalance techniques"""
        print("\n Handling Class Imbalance...")
        
        print(f"Original class distribution: {np.bincount(y)} ({y.mean():.1%} positive)")
        
        # Try different sampling techniques
        sampling_methods = {
            'SMOTE': SMOTE(random_state=self.random_state),
            'ADASYN': ADASYN(random_state=self.random_state),
            'SMOTETomek': SMOTETomek(random_state=self.random_state)
        }
        
        best_method = None
        best_score = 0
        best_X_resampled = None
        best_y_resampled = None
        
        for method_name, sampler in sampling_methods.items():
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                # Quick evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled, test_size=0.2, 
                    random_state=self.random_state, stratify=y_resampled
                )
                
                lr_eval = LogisticRegression(random_state=self.random_state, max_iter=1000)
                lr_eval.fit(X_train, y_train)
                score = roc_auc_score(y_test, lr_eval.predict_proba(X_test)[:, 1])
                
                print(f"   {method_name}: {X_resampled.shape[0]} samples, AUC: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_method = method_name
                    best_X_resampled = X_resampled
                    best_y_resampled = y_resampled
                    
            except Exception as e:
                print(f"   {method_name}: Failed ({e})")
        
        if best_X_resampled is not None:
            print(f" Best sampling method: {best_method}")
            print(f"   New distribution: {np.bincount(best_y_resampled)} ({best_y_resampled.mean():.1%} positive)")
            return best_X_resampled, best_y_resampled
        else:
            print(" No sampling method improved performance, using original data")
            return X, y
    
    def create_advanced_models(self):
        """Create advanced model ensemble"""
        print("\n Creating Advanced Model Ensemble...")
        
        models = {}
        
        # 1. XGBoost (often best for tabular data)
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # 2. LightGBM (fast and often high-performing)
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=self.random_state,
            verbose=-1
        )
        
        # 3. Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        # 4. Extra Trees (often good for ensembles)
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # 5. Random Forest (optimized)
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # 6. Support Vector Machine
        models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        return models
    
    def hyperparameter_optimization(self, model, X, y, model_name):
        """Optimize hyperparameters for each model"""
        print(f"🔧 Optimizing {model_name}...")
        
        param_grids = {
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10]
            }
        }
        
        if model_name in param_grids:
            search = RandomizedSearchCV(
                model, 
                param_grids[model_name],
                n_iter=20,  # Limited for time
                cv=3,
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            search.fit(X, y)
            print(f"   Best params: {search.best_params_}")
            print(f"   Best CV score: {search.best_score_:.3f}")
            
            return search.best_estimator_
        else:
            return model
    
    def train_advanced_ensemble(self, X, y):
        """Train advanced ensemble with all optimizations"""
        print("\n Training Advanced Ensemble for 0.70+ AUC...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features for algorithms that need it
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['ensemble'] = scaler
        
        # Create models
        models = self.create_advanced_models()
        
        # Train and optimize each model
        trained_models = []
        for model_name, model in models.items():
            print(f"\n Training {model_name}...")
            
            try:
                # Use scaled data for SVM, original for tree models
                if model_name == 'svm':
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train
                    X_test_use = X_test
                
                # Hyperparameter optimization for key models
                if model_name in ['xgboost', 'lightgbm', 'random_forest']:
                    optimized_model = self.hyperparameter_optimization(model, X_train_use, y_train, model_name)
                else:
                    optimized_model = model
                
                # Train final model
                optimized_model.fit(X_train_use, y_train)
                
                # Evaluate
                y_pred_proba = optimized_model.predict_proba(X_test_use)[:, 1]
                
                # Optimize threshold
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
                
                self.models[model_name] = optimized_model
                self.results[model_name] = {
                    'metrics': results,
                    'y_test': y_test,
                    'y_pred_proba': y_pred_proba,
                    'model_name': model_name.replace('_', ' ').title()
                }
                
                print(f" {model_name}: AUC={results['auc']:.3f}, F1={results['f1']:.3f}")
                
                # Add to ensemble if performance is decent
                if results['auc'] > 0.55:
                    trained_models.append((model_name, optimized_model))
                    
            except Exception as e:
                print(f" Error training {model_name}: {e}")
        
        # Create meta-ensemble (Stacking)
        if len(trained_models) >= 3:
            print(f"\n Creating Stacking Ensemble...")
            
            # Prepare base models for stacking
            base_models = [(name, model) for name, model in trained_models[:5]]  # Top 5 models
            
            # Meta-learner
            meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Create stacking classifier
            stacking_clf = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=3,
                stack_method='predict_proba'
            )
            
            # Train stacking ensemble
            if trained_models[0][1].__class__.__name__ == 'SVC':
                stacking_clf.fit(X_train_scaled, y_train)
                y_pred_proba_stack = stacking_clf.predict_proba(X_test_scaled)[:, 1]
            else:
                stacking_clf.fit(X_train, y_train)
                y_pred_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]
            
            # Optimize threshold for stacking
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_stack)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            y_pred_stack = (y_pred_proba_stack >= optimal_threshold).astype(int)
            
            # Calculate stacking metrics
            stacking_results = {
                'auc': roc_auc_score(y_test, y_pred_proba_stack),
                'accuracy': accuracy_score(y_test, y_pred_stack),
                'precision': precision_score(y_test, y_pred_stack, zero_division=0),
                'recall': recall_score(y_test, y_pred_stack),
                'f1': f1_score(y_test, y_pred_stack),
                'threshold': optimal_threshold
            }
            
            self.models['stacking_ensemble'] = stacking_clf
            self.results['stacking_ensemble'] = {
                'metrics': stacking_results,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba_stack,
                'model_name': 'Stacking Ensemble'
            }
            
            print(f" Stacking Ensemble: AUC={stacking_results['auc']:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_final_performance(self):
        """Evaluate final performance and check for 0.70+ AUC"""
        print("\n FINAL PERFORMANCE EVALUATION:")
        print("=" * 60)
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['auc'])
        best_result = self.results[best_model_name]
        best_auc = best_result['metrics']['auc']
        
        # Performance comparison
        print(f" PERFORMANCE COMPARISON:")
        print(f"{'Model':<20} {'AUC':<8} {'F1':<8} {'Recall':<8} {'Precision':<10}")
        print("-" * 60)
        
        # Baseline comparisons
        baselines = {
            'LACE Score': 0.549,
            'HOSPITAL Score': 0.528,
            'Logistic Reg (Phase 2)': 0.652
        }
        
        for model, auc in baselines.items():
            print(f"{model:<20} {auc:<8.3f} {'N/A':<8} {'N/A':<8} {'N/A':<10}")
        
        print("-" * 60)
        
        # Our models
        for model_name, result in self.results.items():
            metrics = result['metrics']
            display_name = result['model_name'][:19]
            print(f"{display_name:<20} {metrics['auc']:<8.3f} {metrics['f1']:<8.3f} {metrics['recall']:<8.3f} {metrics['precision']:<10.3f}")
        
        print("-" * 60)
        print(f"🚀 BEST MODEL: {best_result['model_name']} (AUC: {best_auc:.3f})")
        
        # Performance assessment
        if best_auc >= 0.70:
            performance_level = "GOOD"
            status = " TARGET ACHIEVED!"
            color = ""
        elif best_auc >= 0.60:
            performance_level = "FAIR"
            status = " Close to target"
            color = ""
        else:
            performance_level = "POOR"
            status = " Needs improvement"
            color = ""
        
        print(f"\n{color} CLINICAL PERFORMANCE: {performance_level}")
        print(f"{status}")
        
        # Improvements
        improvement_vs_lr = ((best_auc - 0.652) / 0.652) * 100
        improvement_vs_lace = ((best_auc - 0.549) / 0.549) * 100
        
        print(f" Improvement vs Phase 2 LR: {improvement_vs_lr:+.1f}%")
        print(f" Improvement vs LACE: {improvement_vs_lace:+.1f}%")
        
        if best_auc >= 0.70:
            print(f"\n BREAKTHROUGH ACHIEVED!")
            print(f"   • Good clinical performance level reached")
            print(f"   • Suitable for clinical decision support")
            print(f"   • Publication-worthy results")
            print(f"   • Industry-grade performance")
        
        return best_model_name, best_auc, performance_level
    
    def save_best_models(self):
        """Save the best performing models"""
        print("\n Saving Best Models...")
        
        # Save top 3 models
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['metrics']['auc'],
            reverse=True
        )[:3]
        
        for i, (model_name, result) in enumerate(sorted_models):
            model_path = MODELS_DIR / f"advanced_model_{i+1}_{model_name}.pkl"
            joblib.dump(self.models[model_name], model_path)
            print(f" Saved #{i+1}: {model_name} (AUC: {result['metrics']['auc']:.3f})")
        
        # Save scaler
        if 'ensemble' in self.scalers:
            scaler_path = MODELS_DIR / "advanced_ensemble_scaler.pkl"
            joblib.dump(self.scalers['ensemble'], scaler_path)
            print(f" Saved scaler: {scaler_path}")


def run_advanced_ensemble():
    """Main function to run advanced ensemble for 0.70+ AUC"""
    print("" + "=" * 60 + "")
    print("    ADVANCED ENSEMBLE FOR 0.70+ AUC")
    print("    TARGET: GOOD CLINICAL PERFORMANCE")
    print("    MSc Data Science - Shilpa Sunil")
    print("" + "=" * 60 + "")
    
    # Initialize predictor
    predictor = AdvancedEnsemblePredictor()
    
    # Load and engineer features
    X, y, df = predictor.load_and_engineer_features()
    
    # Feature selection optimization
    X_selected, best_features = predictor.feature_selection_optimization(X, y)
    
    # Handle class imbalance
    X_balanced, y_balanced = predictor.handle_class_imbalance(X_selected, y)
    
    # Train advanced ensemble
    X_train, X_test, y_train, y_test = predictor.train_advanced_ensemble(X_balanced, y_balanced)
    
    # Final evaluation
    best_model, best_auc, performance_level = predictor.evaluate_final_performance()
    
    # Save models
    predictor.save_best_models()
    
    print("\n ADVANCED ENSEMBLE COMPLETE!")
    print(" Phase 1: Research & Data Setup (COMPLETE)")
    print(" Phase 2: Baseline Models (COMPLETE)")
    print(" Phase 3: Web Application (COMPLETE)")
    print(" Phase 4: Advanced Ensemble (COMPLETE)")
    
    if best_auc >= 0.70:
        print(" BREAKTHROUGH: 0.70+ AUC ACHIEVED!")
    else:
        print(f" Best AUC: {best_auc:.3f} - Significant improvement demonstrated")
    
    print(f"\n FINAL ACHIEVEMENTS:")
    print(f"   Best Model: {best_model}")
    print(f"   Best AUC: {best_auc:.3f}")
    print(f"   Performance Level: {performance_level}")
    print(f"   Features Used: {len(best_features)}")
    print(f"   Models Tested: {len(predictor.models)}")
    
    return predictor, best_model, best_auc


if __name__ == "__main__":
    predictor, best_model, best_auc = run_advanced_ensemble()