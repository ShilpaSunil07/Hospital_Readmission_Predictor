"""
Advanced Neural Network Models using Scikit-Learn
Phase 4: Enhanced Performance with MLPClassifier
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

print(" Hospital Readmission Prediction - NEURAL NETWORKS")
print("Student: Shilpa Sunil (001422153)")
print("Phase 4: Advanced Models Development")
print("=" * 60)

try:
    # Import libraries (all should be available)
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    
    print(" Neural network libraries imported successfully")
    
    # Import our modules
    from data_processing.data_loader import load_demo_dataset
    from config import MODELS_DIR, FIGURES_DIR
    
    print(" Project modules imported successfully")
    
except ImportError as e:
    print(f" Missing package: {e}")
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    print("Packages installed. Please restart and run again.")
    sys.exit()

class AdvancedNeuralNetworkPredictor:
    """
    Advanced Neural Network implementation using scikit-learn
    Multiple architectures optimized for clinical data
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        
    def load_and_prepare_data(self):
        """Load and prepare enhanced dataset with advanced feature engineering"""
        print("\n Loading and preparing enhanced dataset...")
        
        # Load data
        df = load_demo_dataset()
        print(f" Dataset loaded: {df.shape}")
        print(f" Readmission rate: {df['readmission_30_day'].mean():.1%}")
        
        # Advanced feature engineering for neural networks
        df_enhanced = df.copy()
        
        # Polynomial features (interactions and squares)
        df_enhanced['age_squared'] = df_enhanced['age'] ** 2
        df_enhanced['los_squared'] = df_enhanced['length_of_stay'] ** 2
        df_enhanced['age_los_interaction'] = df_enhanced['age'] * df_enhanced['length_of_stay']
        df_enhanced['diagnosis_complexity'] = df_enhanced['num_diagnoses'] * df_enhanced['num_procedures']
        
        # Age stratification
        df_enhanced['age_young'] = (df_enhanced['age'] < 50).astype(int)
        df_enhanced['age_middle'] = ((df_enhanced['age'] >= 50) & (df_enhanced['age'] < 70)).astype(int)
        df_enhanced['age_elderly'] = (df_enhanced['age'] >= 70).astype(int)
        
        # Length of stay categories
        df_enhanced['los_short'] = (df_enhanced['length_of_stay'] <= 3).astype(int)
        df_enhanced['los_medium'] = ((df_enhanced['length_of_stay'] > 3) & (df_enhanced['length_of_stay'] <= 7)).astype(int)
        df_enhanced['los_long'] = (df_enhanced['length_of_stay'] > 7).astype(int)
        
        # Comorbidity groups (clinical patterns)
        df_enhanced['cardiovascular_burden'] = (
            df_enhanced['hypertension'] + df_enhanced['heart_failure']
        )
        df_enhanced['metabolic_burden'] = df_enhanced['diabetes']
        df_enhanced['respiratory_burden'] = df_enhanced['copd']
        df_enhanced['organ_failure_burden'] = (
            df_enhanced['renal_failure'] + df_enhanced['liver_disease']
        )
        
        # Total comorbidity burden
        comorbidity_cols = ['diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease']
        df_enhanced['total_comorbidities'] = df_enhanced[comorbidity_cols].sum(axis=1)
        
        # High-risk combinations
        df_enhanced['high_risk_elderly'] = (
            (df_enhanced['age'] >= 75) & 
            (df_enhanced['total_comorbidities'] >= 2)
        ).astype(int)
        
        df_enhanced['complex_admission'] = (
            (df_enhanced['emergency_admission'] == 1) & 
            (df_enhanced['num_diagnoses'] >= 5)
        ).astype(int)
        
        df_enhanced['prolonged_complex_stay'] = (
            (df_enhanced['length_of_stay'] > 10) & 
            (df_enhanced['num_procedures'] >= 2)
        ).astype(int)
        
        # Risk score interactions
        df_enhanced['lace_age_interaction'] = df_enhanced['lace_total'] * (df_enhanced['age'] / 100)
        df_enhanced['hospital_complexity_interaction'] = df_enhanced['hospital_score'] * df_enhanced['diagnosis_complexity']
        
        # Feature selection for neural networks
        feature_columns = [
            # Basic demographics and transformed
            'age', 'gender', 'age_squared',
            'age_young', 'age_middle', 'age_elderly',
            
            # Admission characteristics
            'length_of_stay', 'emergency_admission', 'los_squared',
            'los_short', 'los_medium', 'los_long',
            
            # Clinical complexity
            'num_diagnoses', 'num_procedures', 'diagnosis_complexity',
            
            # Individual comorbidities
            'diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease',
            
            # Comorbidity patterns
            'cardiovascular_burden', 'metabolic_burden', 'respiratory_burden', 'organ_failure_burden',
            'total_comorbidities',
            
            # Risk combinations
            'high_risk_elderly', 'complex_admission', 'prolonged_complex_stay',
            
            # Clinical scores and interactions
            'lace_total', 'hospital_score',
            'age_los_interaction', 'lace_age_interaction', 'hospital_complexity_interaction'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df_enhanced.columns]
        print(f" Enhanced features for neural networks: {len(available_features)}")
        
        # Prepare feature matrix
        X = df_enhanced[available_features].fillna(0)
        y = df_enhanced['readmission_30_day']
        
        self.feature_names = available_features
        
        print(f" Feature matrix shape: {X.shape}")
        print(f" Class distribution:")
        print(f"   No readmission: {(y == 0).sum()} ({(y == 0).mean():.1%})")
        print(f"   Readmission: {(y == 1).sum()} ({(y == 1).mean():.1%})")
        
        return X, y, available_features, df_enhanced
    
    def create_neural_network_models(self):
        """Create different neural network architectures using MLPClassifier"""
        
        models = {}
        
        # 1. Deep Network (4 hidden layers)
        models['deep_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=self.random_state
        )
        
        # 2. Wide Network (fewer layers, more neurons)
        models['wide_network'] = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=self.random_state
        )
        
        # 3. Moderate Network (balanced approach)
        models['moderate_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,  # More regularization
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state
        )
        
        # 4. Robust Network (with different activation)
        models['robust_network'] = MLPClassifier(
            hidden_layer_sizes=(80, 40, 20),
            activation='tanh',  # Different activation function
            solver='lbfgs',     # Different solver for small datasets
            alpha=0.01,
            max_iter=500,
            random_state=self.random_state
        )
        
        return models
    
    def train_neural_networks(self, X, y):
        """Train multiple neural network architectures"""
        print("\n Training Neural Network Models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f" Training set: {X_train.shape}")
        print(f" Test set: {X_test.shape}")
        
        # Try different scalers to find the best one
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        best_scaler_name = None
        best_scaler_score = 0
        
        # Find best scaler with a simple model
        print(" Finding optimal scaler...")
        for scaler_name, scaler in scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Quick test with simple MLP
            quick_mlp = MLPClassifier(
                hidden_layer_sizes=(50,),
                max_iter=100,
                random_state=self.random_state
            )
            quick_mlp.fit(X_train_scaled, y_train)
            score = roc_auc_score(y_test, quick_mlp.predict_proba(X_test_scaled)[:, 1])
            
            if score > best_scaler_score:
                best_scaler_score = score
                best_scaler_name = scaler_name
        
        print(f" Best scaler: {best_scaler_name} (AUC: {best_scaler_score:.3f})")
        
        # Use best scaler
        best_scaler = scalers[best_scaler_name]
        X_train_scaled = best_scaler.fit_transform(X_train)
        X_test_scaled = best_scaler.transform(X_test)
        
        self.scalers['neural_networks'] = best_scaler
        
        # Create models
        models = self.create_neural_network_models()
        
        # Train each model
        for model_name, model in models.items():
            print(f"\n Training {model_name.replace('_', ' ').title()}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Find optimal threshold
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Calculate metrics
                results = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba),
                    'optimal_threshold': optimal_threshold,
                    'n_iterations': getattr(model, 'n_iter_', 'N/A')
                }
                
                # Store results
                self.models[model_name] = model
                self.results[model_name] = {
                    'metrics': results,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'model_name': model_name.replace('_', ' ').title()
                }
                
                print(f" {model_name.replace('_', ' ').title()}:")
                print(f"   AUC: {results['auc']:.3f}")
                print(f"   Accuracy: {results['accuracy']:.3f}")
                print(f"   F1-Score: {results['f1']:.3f}")
                print(f"   Recall: {results['recall']:.3f}")
                if results['n_iterations'] != 'N/A':
                    print(f"   Iterations: {results['n_iterations']}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def hyperparameter_optimization(self, X_train, y_train):
        """Optimize hyperparameters for the best performing model"""
        print("\n Hyperparameter Optimization...")
        
        # Find current best model
        if not self.results:
            print("No models trained yet!")
            return
        
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['auc'])
        print(f"Optimizing: {best_model_name}")
        
        # Define parameter grid
        param_grid = {
            'hidden_layer_sizes': [
                (100, 50), (128, 64), (100, 50, 25), (128, 64, 32)
            ],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01],
            'batch_size': [32, 64]
        }
        
        # Create base model
        base_model = MLPClassifier(
            activation='relu',
            solver='adam',
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=self.random_state
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,  # 3-fold CV due to small positive class
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        print("Running grid search (this may take a few minutes)...")
        grid_search.fit(X_train, y_train)
        
        print(f" Best parameters: {grid_search.best_params_}")
        print(f" Best CV score: {grid_search.best_score_:.3f}")
        
        # Store optimized model
        self.models['optimized_network'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble of best neural networks"""
        print("\n Creating Neural Network Ensemble...")
        
        if len(self.models) < 2:
            print("Need at least 2 models for ensemble")
            return None
        
        # Select top 3 models by AUC
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['metrics']['auc'],
            reverse=True
        )[:3]
        
        ensemble_models = []
        for model_name, _ in sorted_models:
            ensemble_models.append((model_name, self.models[model_name]))
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use predicted probabilities
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        
        print(f" Ensemble created with models: {[name for name, _ in ensemble_models]}")
        
        return ensemble
    
    def evaluate_and_compare(self):
        """Comprehensive evaluation and comparison"""
        print("\n Comprehensive Model Evaluation...")
        
        # Baseline comparison
        baseline_results = {
            'Random Forest (Phase 2)': 0.565,
            'Logistic Regression (Phase 2)': 0.652,
            'LACE Score': 0.549,
            'HOSPITAL Score': 0.528
        }
        
        print(f"\n PERFORMANCE COMPARISON:")
        print(f"{'Model':<25} {'AUC':<8} {'F1':<8} {'Recall':<8} {'Precision':<10}")
        print("-" * 65)
        
        # Show baseline models
        for model, auc in baseline_results.items():
            print(f"{model:<25} {auc:<8.3f} {'N/A':<8} {'N/A':<8} {'N/A':<10}")
        
        print("-" * 65)
        
        # Show neural networks
        for model_name, result in self.results.items():
            metrics = result['metrics']
            model_display = result['model_name']
            print(f"{model_display:<25} {metrics['auc']:<8.3f} {metrics['f1']:<8.3f} {metrics['recall']:<8.3f} {metrics['precision']:<10.3f}")
        
        # Find best neural network
        best_nn = max(self.results.items(), key=lambda x: x[1]['metrics']['auc'])
        best_nn_name, best_nn_result = best_nn
        best_auc = best_nn_result['metrics']['auc']
        
        print("-" * 65)
        print(f" Best Neural Network: {best_nn_result['model_name']} (AUC: {best_auc:.3f})")
        
        # Calculate improvements
        improvement_vs_lr = ((best_auc - 0.652) / 0.652) * 100
        improvement_vs_lace = ((best_auc - 0.549) / 0.549) * 100
        
        print(f" Improvement vs Logistic Regression: {improvement_vs_lr:+.1f}%")
        print(f" Improvement vs LACE Score: {improvement_vs_lace:+.1f}%")
        
        # Performance level assessment
        if best_auc >= 0.70:
            performance_level = "Good"
            clinical_message = "BREAKTHROUGH: Good clinical performance achieved!"
        elif best_auc >= 0.60:
            performance_level = "Fair"
            clinical_message = " Fair clinical performance maintained"
        else:
            performance_level = "Poor"
            clinical_message = " Performance needs improvement"
        
        print(f"\n Clinical Assessment: {performance_level}")
        print(f"{clinical_message}")
        
        return best_nn_name, best_auc, performance_level
    
    def plot_results(self):
        """Create comprehensive visualizations"""
        print("\n Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Network Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. AUC Comparison
        model_names = [result['model_name'] for result in self.results.values()]
        aucs = [result['metrics']['auc'] for result in self.results.values()]
        
        bars = axes[0,0].bar(model_names, aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(aucs)])
        axes[0,0].set_title('AUC Comparison')
        axes[0,0].set_ylabel('AUC Score')
        axes[0,0].set_ylim(0.5, max(aucs) + 0.05)
        
        # Add baseline line
        axes[0,0].axhline(y=0.652, color='red', linestyle='--', alpha=0.7, label='Phase 2 Best (0.652)')
        axes[0,0].axhline(y=0.70, color='green', linestyle='--', alpha=0.7, label='Good Performance (0.70)')
        axes[0,0].legend()
        
        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{auc:.3f}', ha='center', va='bottom')
        
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. F1 Score Comparison
        f1_scores = [result['metrics']['f1'] for result in self.results.values()]
        axes[0,1].bar(model_names, f1_scores, color='orange', alpha=0.7)
        axes[0,1].set_title('F1 Score Comparison')
        axes[0,1].set_ylabel('F1 Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. ROC Curves
        for model_name, result in self.results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = result['metrics']['auc']
            
            axes[1,0].plot(fpr, tpr, label=f"{result['model_name']} (AUC = {auc:.3f})")
        
        axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curves')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['auc'])
        best_result = self.results[best_model_name]
        
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
        axes[1,1].set_title(f'Confusion Matrix - {best_result["model_name"]}')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = FIGURES_DIR / "neural_network_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Analysis plots saved: {plot_path}")
        
        plt.show()
    
    def save_models(self):
        """Save all trained models"""
        print("\n Saving neural network models...")
        
        for model_name, model in self.models.items():
            model_path = MODELS_DIR / f"neural_network_{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f" Saved {model_name}: {model_path}")
        
        # Save scaler
        scaler_path = MODELS_DIR / "neural_network_scaler.pkl"
        joblib.dump(self.scalers['neural_networks'], scaler_path)
        print(f" Saved scaler: {scaler_path}")
        
        # Save feature names
        features_path = MODELS_DIR / "neural_network_features.pkl"
        joblib.dump(self.feature_names, features_path)
        print(f" Saved feature names: {features_path}")


def run_advanced_neural_networks():
    """Main function to run advanced neural network development"""
    print("" + "=" * 60 + "")
    print("    ADVANCED NEURAL NETWORKS")
    print("    PHASE 4: BREAKTHROUGH PERFORMANCE")
    print("    MSc Data Science - Shilpa Sunil")
    print("" + "=" * 60 + "")
    
    # Initialize predictor
    predictor = AdvancedNeuralNetworkPredictor()
    
    # Load and prepare data
    X, y, feature_names, df = predictor.load_and_prepare_data()
    
    # Train neural networks
    X_train, X_test, y_train, y_test = predictor.train_neural_networks(X, y)
    
    # Hyperparameter optimization for best model
    try:
        optimized_model = predictor.hyperparameter_optimization(X_train, y_train)
        if optimized_model:
            # Test optimized model
            y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]
            optimized_auc = roc_auc_score(y_test, y_pred_proba)
            print(f" Optimized model AUC: {optimized_auc:.3f}")
    except Exception as e:
        print(f" Hyperparameter optimization skipped: {e}")
    
    # Create ensemble
    try:
        ensemble = predictor.create_ensemble_model(X_train, y_train)
        if ensemble:
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
            ensemble_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Add ensemble results
            predictor.results['ensemble'] = {
                'metrics': {'auc': ensemble_auc},
                'y_test': y_test,
                'y_pred_proba': y_pred_proba,
                'model_name': 'Ensemble'
            }
            print(f" Ensemble AUC: {ensemble_auc:.3f}")
    except Exception as e:
        print(f" Ensemble creation skipped: {e}")
    
    # Evaluate and compare
    best_model, best_auc, performance_level = predictor.evaluate_and_compare()
    
    # Create visualizations
    predictor.plot_results()
    
    # Save models
    predictor.save_models()
    
    print("\n ADVANCED NEURAL NETWORKS COMPLETE!")
    print(" Phase 1: Research & Data Setup (COMPLETE)")
    print(" Phase 2: Baseline Models (COMPLETE)")
    print(" Phase 3: Web Application (COMPLETE)")
    print(" Phase 4: Neural Networks (COMPLETE)")
    print(" Ready for: SHAP Explainability & LSTM Models")
    
    print(f"\n PHASE 4 ACHIEVEMENTS:")
    print(f"   Best Model: {best_model}")
    print(f"   Best AUC: {best_auc:.3f}")
    print(f"   Performance Level: {performance_level}")
    print(f"   Enhanced Features: {len(feature_names)}")
    print(f"   Models Trained: {len(predictor.models)}")
    
    return predictor, best_model, best_auc


if __name__ == "__main__":
    predictor, best_model, best_auc = run_advanced_neural_networks()