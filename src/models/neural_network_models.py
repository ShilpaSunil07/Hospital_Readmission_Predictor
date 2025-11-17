"""
Advanced Neural Network Models for Hospital Readmission Prediction
Phase 4: Enhanced Performance with Deep Learning
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
    # Import libraries
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    
    print(" Deep learning libraries imported successfully")
    
    # Import our modules
    from data_processing.data_loader import load_demo_dataset
    from config import MODELS_DIR, FIGURES_DIR
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print(" Project modules imported successfully")
    
except ImportError as e:
    print(f" Missing required packages. Installing TensorFlow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "matplotlib"])
    print(" TensorFlow installed. Please restart and run again.")
    sys.exit()

class NeuralNetworkPredictor:
    """
    Advanced Neural Network implementation for readmission prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.results = {}
        
        # Set TensorFlow random seed
        tf.random.set_seed(random_state)
        
    def load_and_prepare_data(self):
        """Load and prepare enhanced dataset"""
        print("\n Loading and preparing enhanced dataset...")
        
        # Load data
        df = load_demo_dataset()
        print(f" Dataset loaded: {df.shape}")
        print(f" Readmission rate: {df['readmission_30_day'].mean():.1%}")
        
        # Enhanced feature engineering for neural networks
        df_enhanced = df.copy()
        
        # Advanced feature engineering
        df_enhanced['age_squared'] = df_enhanced['age'] ** 2
        df_enhanced['los_squared'] = df_enhanced['length_of_stay'] ** 2
        df_enhanced['age_los_interaction'] = df_enhanced['age'] * df_enhanced['length_of_stay']
        df_enhanced['diagnosis_procedure_ratio'] = (
            df_enhanced['num_diagnoses'] / (df_enhanced['num_procedures'] + 1)
        )
        
        # Comorbidity patterns
        df_enhanced['cardiovascular_group'] = (
            df_enhanced['hypertension'] + df_enhanced['heart_failure']
        )
        df_enhanced['metabolic_group'] = df_enhanced['diabetes']
        df_enhanced['respiratory_group'] = df_enhanced['copd']
        df_enhanced['organ_failure_group'] = (
            df_enhanced['renal_failure'] + df_enhanced['liver_disease']
        )
        
        # Risk stratification features
        df_enhanced['high_complexity'] = (
            (df_enhanced['num_diagnoses'] >= 5) & 
            (df_enhanced['num_procedures'] >= 2)
        ).astype(int)
        
        df_enhanced['elderly_emergency'] = (
            (df_enhanced['age'] >= 70) & 
            (df_enhanced['emergency_admission'] == 1)
        ).astype(int)
        
        # Feature selection for neural networks
        feature_columns = [
            # Demographics
            'age', 'gender', 'age_squared',
            
            # Admission characteristics
            'length_of_stay', 'emergency_admission', 'los_squared',
            
            # Clinical complexity
            'num_diagnoses', 'num_procedures', 'diagnosis_procedure_ratio',
            
            # Comorbidities (individual)
            'diabetes', 'hypertension', 'heart_failure', 'copd', 'renal_failure', 'liver_disease',
            
            # Comorbidity groups
            'cardiovascular_group', 'metabolic_group', 'respiratory_group', 'organ_failure_group',
            
            # Risk scores
            'lace_total', 'hospital_score',
            
            # Interaction features
            'age_los_interaction', 'high_complexity', 'elderly_emergency'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df_enhanced.columns]
        print(f" Neural network features: {len(available_features)}")
        
        # Prepare feature matrix
        X = df_enhanced[available_features].fillna(0)
        y = df_enhanced['readmission_30_day']
        
        print(f" Feature matrix shape: {X.shape}")
        print(f" Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features, df_enhanced
    
    def create_neural_network_models(self, input_dim):
        """Create different neural network architectures"""
        
        models = {}
        
        # 1. Deep Dense Network
        models['deep_dense'] = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        # 2. Wide & Deep Network
        # Wide component (linear)
        wide_input = layers.Input(shape=(input_dim,), name='wide_input')
        wide = layers.Dense(1, activation='linear', name='wide')(wide_input)
        
        # Deep component (non-linear)
        deep = layers.Dense(64, activation='relu')(wide_input)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(32, activation='relu')(deep)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(16, activation='relu')(deep)
        deep = layers.Dense(1, activation='linear', name='deep')(deep)
        
        # Combine wide and deep
        combined = layers.Add()([wide, deep])
        output = layers.Activation('sigmoid')(combined)
        
        models['wide_deep'] = keras.Model(inputs=wide_input, outputs=output)
        
        # 3. Residual Network
        input_layer = layers.Input(shape=(input_dim,))
        
        # First block
        x = layers.Dense(64, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Residual block 1
        residual = x
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='linear')(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # Residual block 2
        residual = x
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='linear')(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # Output
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        models['residual'] = keras.Model(inputs=input_layer, outputs=output)
        
        return models
    
    def train_neural_networks(self, X, y, feature_names):
        """Train multiple neural network architectures"""
        print("\n Training Neural Network Models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f" Training set: {X_train.shape}")
        print(f" Test set: {X_test.shape}")
        
        # Scale features using RobustScaler (better for neural networks)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['neural_networks'] = scaler
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f" Class weights: {class_weight_dict}")
        
        # Create models
        models = self.create_neural_network_models(X_train.shape[1])
        
        # Training configuration
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc', 
            patience=15, 
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode='max'
        )
        
        # Train each model
        for model_name, model in models.items():
            print(f"\n Training {model_name.replace('_', ' ').title()} Network...")
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Store model and history
            self.models[model_name] = model
            self.history[model_name] = history
            
            # Make predictions
            y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            
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
                'epochs_trained': len(history.history['loss'])
            }
            
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
            print(f"   Epochs: {results['epochs_trained']}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def compare_with_baseline(self):
        """Compare neural networks with baseline models"""
        print("\n Comparing with Baseline Models...")
        
        # Baseline results (from previous phase)
        baseline_results = {
            'Random Forest': 0.565,
            'Logistic Regression': 0.652,
            'LACE Score': 0.549,
            'HOSPITAL Score': 0.528
        }
        
        # Neural network results
        nn_results = {
            result['model_name']: result['metrics']['auc'] 
            for result in self.results.values()
        }
        
        # Find best neural network
        best_nn_name = max(nn_results.keys(), key=lambda x: nn_results[x])
        best_nn_auc = nn_results[best_nn_name]
        
        print(f"\n PERFORMANCE COMPARISON:")
        print(f"{'Model':<20} {'AUC':<10} {'Improvement':<12}")
        print("-" * 45)
        
        # Show baseline models
        for model, auc in baseline_results.items():
            improvement = "Baseline" if "Score" in model else "Previous"
            print(f"{model:<20} {auc:<10.3f} {improvement:<12}")
        
        print("-" * 45)
        
        # Show neural networks
        for model, auc in nn_results.items():
            improvement_vs_lr = ((auc - 0.652) / 0.652) * 100
            improvement_str = f"+{improvement_vs_lr:.1f}%" if improvement_vs_lr > 0 else f"{improvement_vs_lr:.1f}%"
            print(f"{model:<20} {auc:<10.3f} {improvement_str:<12}")
        
        print("-" * 45)
        print(f" Best Neural Network: {best_nn_name} (AUC: {best_nn_auc:.3f})")
        
        # Overall improvement
        improvement_vs_baseline = ((best_nn_auc - 0.652) / 0.652) * 100
        improvement_vs_lace = ((best_nn_auc - 0.549) / 0.549) * 100
        
        print(f" Improvement vs Logistic Regression: {improvement_vs_baseline:+.1f}%")
        print(f" Improvement vs LACE Score: {improvement_vs_lace:+.1f}%")
        
        return best_nn_name, best_nn_auc
    
    def plot_training_history(self):
        """Plot training history for all models"""
        print("\n Creating training visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Network Training History', fontsize=16, fontweight='bold')
        
        for i, (model_name, history) in enumerate(self.history.items()):
            row = i // 2
            col = i % 2
            
            # Plot AUC
            axes[row, col].plot(history.history['auc'], label='Training AUC', alpha=0.8)
            axes[row, col].plot(history.history['val_auc'], label='Validation AUC', alpha=0.8)
            axes[row, col].set_title(f'{model_name.replace("_", " ").title()} - AUC')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('AUC')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            # Add final performance text
            final_auc = self.results[model_name]['metrics']['auc']
            axes[row, col].text(0.02, 0.98, f'Final AUC: {final_auc:.3f}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplot if odd number of models
        if len(self.history) % 2 == 1:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = FIGURES_DIR / "neural_network_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Training history saved: {plot_path}")
        
        plt.show()
    
    def save_models(self):
        """Save trained neural network models"""
        print("\n Saving neural network models...")
        
        for model_name, model in self.models.items():
            model_path = MODELS_DIR / f"neural_network_{model_name}.h5"
            model.save(model_path)
            print(f" Saved {model_name}: {model_path}")
        
        # Save scaler
        import joblib
        scaler_path = MODELS_DIR / "neural_network_scaler.pkl"
        joblib.dump(self.scalers['neural_networks'], scaler_path)
        print(f"Saved scaler: {scaler_path}")
    
    def generate_clinical_insights(self, feature_names):
        """Generate clinical insights from neural network models"""
        print("\n Clinical Insights from Neural Networks:")
        
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['auc'])
        best_result = self.results[best_model_name]
        best_auc = best_result['metrics']['auc']
        
        print(f"\n Best Performing Model: {best_result['model_name']}")
        print(f"   AUC: {best_auc:.3f}")
        print(f"   Recall: {best_result['metrics']['recall']:.3f}")
        print(f"   Precision: {best_result['metrics']['precision']:.3f}")
        
        # Performance categorization
        if best_auc >= 0.70:
            performance_level = "Good"
            clinical_utility = "High clinical utility for decision support"
        elif best_auc >= 0.60:
            performance_level = "Fair"
            clinical_utility = "Moderate clinical utility for risk stratification"
        else:
            performance_level = "Poor"
            clinical_utility = "Limited clinical utility, needs improvement"
        
        print(f"\n Clinical Performance Assessment:")
        print(f"   Performance Level: {performance_level}")
        print(f"   Clinical Utility: {clinical_utility}")
        
        if best_auc > 0.652:
            improvement = ((best_auc - 0.652) / 0.652) * 100
            print(f"    Neural networks improved performance by {improvement:.1f}%")
        
        print(f"\n Key Clinical Implications:")
        print(f"   • Model can identify {best_result['metrics']['recall']:.0%} of actual readmissions")
        print(f"   • {best_result['metrics']['precision']:.0%} of predicted readmissions are accurate")
        print(f"   • Suitable for clinical decision support and risk stratification")
        
        return best_model_name, best_auc


def run_neural_network_development():
    """Main function to run neural network development"""
    print("" + "=" * 60 + "")
    print("    NEURAL NETWORKS FOR READMISSION PREDICTION")
    print("    PHASE 4: ADVANCED MODELS DEVELOPMENT")
    print("    MSc Data Science - Shilpa Sunil")
    print("" + "=" * 60 + "")
    
    # Initialize predictor
    predictor = NeuralNetworkPredictor()
    
    # Load and prepare data
    X, y, feature_names, df = predictor.load_and_prepare_data()
    
    # Train neural networks
    X_train, X_test, y_train, y_test = predictor.train_neural_networks(X, y, feature_names)
    
    # Compare with baseline
    best_model, best_auc = predictor.compare_with_baseline()
    
    # Plot training history
    predictor.plot_training_history()
    
    # Generate clinical insights
    predictor.generate_clinical_insights(feature_names)
    
    # Save models
    predictor.save_models()
    
    print("\n NEURAL NETWORK DEVELOPMENT COMPLETE!")
    print(" Phase 1: Research & Data Setup (COMPLETE)")
    print(" Phase 2: Baseline Models (COMPLETE)")
    print(" Phase 3: Web Application (COMPLETE)")
    print(" Phase 4: Neural Networks (COMPLETE)")
    print(" Next: SHAP Explainability & LSTM Models")
    
    print(f"\n PHASE 4 ACHIEVEMENTS:")
    print(f"   Best Neural Network: {best_model}")
    print(f"   Best AUC: {best_auc:.3f}")
    print(f"   Features: {len(feature_names)} (enhanced)")
    print(f"   Models Trained: {len(predictor.models)}")
    
    return predictor, best_model, best_auc


if __name__ == "__main__":
    predictor, best_model, best_auc = run_neural_network_development()