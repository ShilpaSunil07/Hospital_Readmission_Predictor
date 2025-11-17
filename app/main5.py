import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import warnings
from sklearn.metrics import roc_auc_score, classification_report
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title=" Hospital Readmission Predictor",
    layout="wide"
)

# Custom CSS for professional healthcare interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 6px solid #2e86ab;
    }
    .risk-low { border-left-color: #28a745; }
    .risk-moderate { border-left-color: #ffc107; }
    .risk-high { border-left-color: #dc3545; }
    .calibration-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load ML model function
@st.cache_resource
def load_ml_model():
    """Load the trained high-accuracy model"""
    try:
        model_path = "models/high_accuracy_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, True, None
        else:
            model_path = "models/original_optimized_pipeline.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model, True, None
            else:
                return None, False, "No model files found"
    except Exception as e:
        return None, False, str(e)

# Function to calculate actual model performance
@st.cache_data
def calculate_model_performance():
    """Calculate actual model performance metrics"""
    try:
        model, loaded, error = load_ml_model()
        if not loaded:
            return None, None, None, None, error

        data_path = "realistic_hospital_data.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)

            X_transformed = pd.DataFrame()
            X_transformed['age'] = data['age']
            X_transformed['gender_Male'] = (data['gender'] == 1).astype(int)
            X_transformed['admission_type_Emergency'] = data['emergency_admission'].astype(int)
            X_transformed['admission_type_Urgent'] = 0
            X_transformed['length_of_stay'] = data['length_of_stay']
            X_transformed['num_diagnoses'] = data['num_diagnoses']
            X_transformed['num_procedures'] = data['num_procedures']
            X_transformed['diabetes'] = data['diabetes']
            X_transformed['heart_failure'] = data['heart_failure']
            X_transformed['kidney_disease'] = data['renal_failure']
            X_transformed['liver_disease'] = data['liver_disease']

            X_transformed['condition_count'] = (
                data['diabetes'] + data['heart_failure'] +
                data['renal_failure'] + data['liver_disease'] +
                data['hypertension'] + data['copd']
            )
            X_transformed['age_squared'] = data['age'] ** 2
            X_transformed['elderly'] = (data['age'] > 65).astype(int)
            X_transformed['very_elderly'] = (data['age'] > 80).astype(int)
            X_transformed['young'] = (data['age'] < 40).astype(int)

            X_transformed['complexity_score'] = data['num_diagnoses'] + data['num_procedures'] + X_transformed['condition_count']
            X_transformed['high_complexity'] = (X_transformed['complexity_score'] > 5).astype(int)

            X_transformed['age_emergency'] = data['age'] * data['emergency_admission']
            X_transformed['age_conditions'] = data['age'] * X_transformed['condition_count']
            X_transformed['elderly_emergency'] = ((data['age'] > 65) & (data['emergency_admission'] == 1)).astype(int)

            X_transformed['diabetes_heart'] = data['diabetes'] * data['heart_failure']
            X_transformed['diabetes_kidney'] = data['diabetes'] * data['renal_failure']
            X_transformed['multiple_conditions'] = (X_transformed['condition_count'] > 2).astype(int)

            X_transformed['long_stay'] = (data['length_of_stay'] > 7).astype(int)
            X_transformed['short_stay'] = (data['length_of_stay'] <= 3).astype(int)

            X_transformed['high_risk'] = (
                (data['age'] > 70) |
                (X_transformed['condition_count'] > 3) |
                (X_transformed['complexity_score'] > 7) |
                (data['emergency_admission'] == 1)
            ).astype(int)

            X_transformed['feature_22'] = 0
            X_transformed['feature_23'] = 0
            X_transformed['feature_24'] = 0
            X_transformed['feature_25'] = 0

            y = data['readmission_30_day'].copy()
            X_transformed = X_transformed.fillna(0)
            y = y.fillna(0)

            try:
                y_pred_proba = model.predict_proba(X_transformed)[:, 1]
                y_pred = model.predict(X_transformed)
                auc_score = roc_auc_score(y, y_pred_proba)
                report = classification_report(y, y_pred, output_dict=True)
                precision = report['weighted avg']['precision']
                recall = report['weighted avg']['recall']
                f1_score = report['weighted avg']['f1-score']
                return auc_score, precision, recall, f1_score, None
            except Exception as e:
                return None, None, None, None, f"Prediction error: {str(e)}"
        else:
            return None, None, None, None, "No validation data found"

    except Exception as e:
        return None, None, None, None, f"Performance calculation error: {str(e)}"

# Get model feature names
@st.cache_data
def get_model_features():
    model, loaded, _ = load_ml_model()
    if loaded:
        try:
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
            elif hasattr(model, 'named_steps'):
                classifier = model.named_steps.get('classifier', model)
                if hasattr(classifier, 'feature_names_in_'):
                    return list(classifier.feature_names_in_)
            return [
                'age', 'gender_Male', 'admission_type_Emergency', 'admission_type_Urgent',
                'length_of_stay', 'num_diagnoses', 'num_procedures', 'diabetes',
                'heart_failure', 'kidney_disease', 'liver_disease', 'condition_count',
                'age_squared', 'elderly', 'very_elderly', 'young', 'complexity_score',
                'high_complexity', 'age_emergency', 'age_conditions', 'elderly_emergency',
                'diabetes_heart', 'diabetes_kidney', 'multiple_conditions', 'long_stay',
                'short_stay', 'high_risk', 'feature_22', 'feature_23', 'feature_24',
                'feature_25'
            ]
        except:
            return [
                'age', 'gender_Male', 'admission_type_Emergency', 'admission_type_Urgent',
                'length_of_stay', 'num_diagnoses', 'num_procedures', 'diabetes',
                'heart_failure', 'kidney_disease', 'liver_disease', 'condition_count',
                'age_squared', 'elderly', 'very_elderly', 'young', 'complexity_score',
                'high_complexity', 'age_emergency', 'age_conditions', 'elderly_emergency',
                'diabetes_heart', 'diabetes_kidney', 'multiple_conditions', 'long_stay',
                'short_stay', 'high_risk', 'feature_22', 'feature_23', 'feature_24',
                'feature_25'
            ]
    return []

# Create feature vector with exact feature names
def create_exact_feature_vector(age, gender, admission_type, length_of_stay, num_diagnoses, num_procedures, conditions):
    features = {}
    features['age'] = float(age)
    features['gender_Male'] = 1.0 if gender == "Male" else 0.0
    features['admission_type_Emergency'] = 1.0 if admission_type == "Emergency" else 0.0
    features['admission_type_Urgent'] = 1.0 if admission_type == "Urgent" else 0.0
    features['length_of_stay'] = float(length_of_stay)
    features['num_diagnoses'] = float(num_diagnoses)
    features['num_procedures'] = float(num_procedures)
    features['diabetes'] = 1.0 if 'Diabetes' in conditions else 0.0
    features['heart_failure'] = 1.0 if 'Heart Failure' in conditions else 0.0
    features['kidney_disease'] = 1.0 if 'Kidney Disease' in conditions else 0.0
    features['liver_disease'] = 1.0 if 'Liver Disease' in conditions else 0.0

    features['condition_count'] = float(len(conditions))
    features['age_squared'] = float(age ** 2)
    features['elderly'] = 1.0 if age > 65 else 0.0
    features['very_elderly'] = 1.0 if age > 80 else 0.0
    features['young'] = 1.0 if age < 40 else 0.0

    features['complexity_score'] = float(num_diagnoses + num_procedures + len(conditions))
    features['high_complexity'] = 1.0 if (num_diagnoses + num_procedures + len(conditions)) > 5 else 0.0

    features['age_emergency'] = float(age) if admission_type == "Emergency" else 0.0
    features['age_conditions'] = float(age * len(conditions))
    features['elderly_emergency'] = 1.0 if (age > 65 and admission_type == "Emergency") else 0.0

    features['diabetes_heart'] = 1.0 if ('Diabetes' in conditions and 'Heart Failure' in conditions) else 0.0
    features['diabetes_kidney'] = 1.0 if ('Diabetes' in conditions and 'Kidney Disease' in conditions) else 0.0
    features['multiple_conditions'] = 1.0 if len(conditions) > 2 else 0.0

    features['long_stay'] = 1.0 if length_of_stay > 7 else 0.0
    features['short_stay'] = 1.0 if length_of_stay <= 3 else 0.0

    features['high_risk'] = 1.0 if (
        age > 70 or 
        len(conditions) > 3 or 
        (num_diagnoses + num_procedures + len(conditions)) > 7 or
        admission_type == "Emergency"
    ) else 0.0

    features['feature_22'] = 0.0
    features['feature_23'] = 0.0
    features['feature_24'] = 0.0
    features['feature_25'] = 0.0

    feature_names = [
        'age', 'gender_Male', 'admission_type_Emergency', 'admission_type_Urgent',
        'length_of_stay', 'num_diagnoses', 'num_procedures', 'diabetes',
        'heart_failure', 'kidney_disease', 'liver_disease', 'condition_count',
        'age_squared', 'elderly', 'very_elderly', 'young', 'complexity_score',
        'high_complexity', 'age_emergency', 'age_conditions', 'elderly_emergency',
        'diabetes_heart', 'diabetes_kidney', 'multiple_conditions', 'long_stay',
        'short_stay', 'high_risk', 'feature_22', 'feature_23', 'feature_24',
        'feature_25'
    ]
    return pd.DataFrame([features])[feature_names]

# LACE helper 
def compute_lace(length_of_stay, admission_type, conditions, ed_visits_6m=0):
    """
    LACE index = Length of stay + Acuity of admission + Comorbidity + ED visits (6 months)
    """
    if length_of_stay >= 14:
        l_points = 7
    elif length_of_stay >= 7:
        l_points = 5
    elif length_of_stay >= 4:
        l_points = 4
    elif length_of_stay == 3:
        l_points = 3
    elif length_of_stay == 2:
        l_points = 2
    elif length_of_stay == 1:
        l_points = 1
    else:
        l_points = 0

    a_points = 3 if admission_type == "Emergency" else 0

    c_points = 0
    if "Heart Failure" in conditions:
        c_points += 1
    if "Diabetes" in conditions:
        c_points += 1
    if "Kidney Disease" in conditions:
        c_points += 2
    if "Liver Disease" in conditions:
        c_points += 1
    c_points = min(c_points, 5)

    if ed_visits_6m >= 4:
        e_points = 4
    elif ed_visits_6m == 3:
        e_points = 3
    elif ed_visits_6m == 2:
        e_points = 2
    elif ed_visits_6m == 1:
        e_points = 1
    else:
        e_points = 0

    total = l_points + a_points + c_points + e_points
    if total >= 10:
        band = "High"
    elif total >= 5:
        band = "Moderate"
    else:
        band = "Low"
    return total, band, {"L_points": l_points, "A_points": a_points, "C_points": c_points, "E_points": e_points}

# PROPER Clinical Calibration Function
def clinical_calibration(raw_ml_prediction, age, admission_type, conditions, length_of_stay, num_diagnoses, num_procedures):
    if age < 30:
        base_clinical_risk = 0.05
    elif age < 50:
        base_clinical_risk = 0.07
    elif age < 70:
        base_clinical_risk = 0.10
    else:
        base_clinical_risk = 0.14

    condition_multiplier = 1.0
    if 'Heart Failure' in conditions:
        condition_multiplier *= 1.6
    if 'Kidney Disease' in conditions:
        condition_multiplier *= 1.4
    if 'Diabetes' in conditions:
        condition_multiplier *= 1.2
    if 'Liver Disease' in conditions:
        condition_multiplier *= 1.3

    admission_multiplier = 1.0
    if admission_type == "Emergency":
        admission_multiplier = 1.3
    elif admission_type == "Urgent":
        admission_multiplier = 1.1

    complexity_multiplier = 1.0 + (num_diagnoses - 1) * 0.05 + num_procedures * 0.03
    if length_of_stay > 7:
        complexity_multiplier *= 1.1

    expected_clinical_risk = base_clinical_risk * condition_multiplier * admission_multiplier * complexity_multiplier
    expected_clinical_risk = min(expected_clinical_risk, 0.40)

    ml_factor = raw_ml_prediction / 0.20
    ml_factor = max(0.5, min(2.0, ml_factor))

    final_risk = expected_clinical_risk * ml_factor
    if final_risk <= 0.03:
        final_risk = 0.03
    elif final_risk <= 0.35:
        pass
    elif final_risk <= 0.50:
        excess = final_risk - 0.35
        final_risk = 0.35 + (excess * 0.7)
    else:
        final_risk = 0.35 + ((final_risk - 0.35) * 0.3)
        final_risk = min(final_risk, 0.50)

    return final_risk, {
        'raw_ml': raw_ml_prediction,
        'base_clinical': base_clinical_risk,
        'condition_multiplier': condition_multiplier,
        'admission_multiplier': admission_multiplier,
        'complexity_multiplier': complexity_multiplier,
        'expected_clinical': expected_clinical_risk,
        'ml_factor': ml_factor,
        'final': final_risk
    }

# Header
def render_header():
    auc_score, precision, recall, f1_score, error = calculate_model_performance()
    if auc_score is not None:
        st.markdown(f"""
        <div class="main-header">
            <h1> Hospital Readmission Predictor</h1>
            <p>Random Forest Model  with Proper Clinical Calibration</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="main-header">
            <h1> Hospital Readmission Predictor</h1>
            <p>Random Forest Model with Proper Clinical Calibration</p>
        </div>
        """, unsafe_allow_html=True)
        if error:
            st.warning(f" Performance metrics unavailable: {error}")

# Sidebar
with st.sidebar:
    st.markdown("### ML Model Status")
    model, model_loaded, error_msg = load_ml_model()
    if model_loaded:
        st.success(" Random Forest Model Loaded")
        st.success(" Clinical Calibration Active")

        expected_features = get_model_features()
        st.markdown(f"### Model Features ({len(expected_features)})")
        with st.expander("View Feature List"):
            for i, feature in enumerate(expected_features, 1):
                st.text(f"{i}. {feature}")
    else:
        st.error(" ML Model Error")
        if error_msg:
            st.error(f"Error: {error_msg}")

    st.markdown("### Model Information")
    auc_score, precision, recall, f1_score, error = calculate_model_performance()
    if auc_score is not None:
        st.markdown(f"""
        **Algorithm:** Random Forest  
        **Training Data:** MIMIC-III schema derived + Realistic Data  
        **Features:** {len(get_model_features())} optimized features  
        **Calibration:** Evidence-Based Clinical
        """)
    else:
        st.markdown(f"""
        **Algorithm:** Random Forest  
        **Training Data:** MIMIC-III + Realistic Data  
        **Features:** {len(get_model_features())} optimized features  
        **Performance:** Calculating...  
        **Calibration:** Evidence-Based Clinical
        """)
        if error:
            st.error(f"Error: {error}")

# Render header
render_header()

# Main UI
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Patient Information")

    st.markdown("#### Demographics")
    age = st.slider("Age (years)", 18, 100, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("#### Admission Details")
    admission_type = st.selectbox("Admission Type", ["Emergency", "Elective"])
    length_of_stay = st.slider("Length of Stay (days)", 1, 30, 3)
    ed_visits_6m = st.number_input("ED Visits in Last 6 Months (optional)", min_value=0, max_value=10, value=0, step=1)

    st.markdown("#### Clinical Details")
    num_diagnoses = st.slider("Number of Diagnoses", 1, 15, 2)
    num_procedures = st.slider("Number of Procedures", 0, 10, 1)

    st.markdown("#### Medical Conditions")
    conditions = []
    if st.checkbox("Heart Failure"):
        conditions.append("Heart Failure")
    if st.checkbox("Kidney Disease"):
        conditions.append("Kidney Disease")
    if st.checkbox("Diabetes"):
        conditions.append("Diabetes")
    if st.checkbox("Liver Disease"):
        conditions.append("Liver Disease")

    predict_button = st.button(" ML Prediction", use_container_width=True, type="primary")

with col2:
    st.markdown("###  ML Prediction Results")

    if predict_button:
        if model_loaded:
            try:
                feature_vector = create_exact_feature_vector(
                    age, gender, admission_type, length_of_stay,
                    num_diagnoses, num_procedures, conditions
                )

                raw_ml_prediction = model.predict_proba(feature_vector)[0][1]

                final_prediction, calibration_details = clinical_calibration(
                    raw_ml_prediction, age, admission_type, conditions,
                    length_of_stay, num_diagnoses, num_procedures
                )

                # Risk categorization
                if final_prediction < 0.10:
                    risk_level = "Low Risk"
                    risk_color = "#28a745"
                    risk_class = "risk-low"
                elif final_prediction < 0.20:
                    risk_level = "Moderate Risk"
                    risk_color = "#ffc107"
                    risk_class = "risk-moderate"
                else:
                    risk_level = "High Risk"
                    risk_color = "#dc3545"
                    risk_class = "risk-high"

                # LACE (computed BEFORE card so we can place it inside)
                lace_score, lace_band, lace_parts = compute_lace(
                    length_of_stay=length_of_stay,
                    admission_type=admission_type,
                    conditions=conditions,
                    ed_visits_6m=ed_visits_6m
                )

                # Main result card WITH LACE inside (no duplicate row below)
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h2 style="color: {risk_color}; margin: 0;">{risk_level}</h2>
                    <h1 style="color: {risk_color}; margin: 0.5rem 0;">{final_prediction:.1%}</h1>
                    <p style="margin: 0;"><strong>30-Day Readmission Probability</strong></p>
                    <p style="margin: 0.35rem 0; font-size: 1.1em;">
                        <strong>LACE Score:</strong> {lace_score} ({lace_band})
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Optional: components breakdown
                with st.expander("See LACE components"):
                    st.write(f"**L (Length of Stay) points:** {lace_parts['L_points']}")
                    st.write(f"**A (Acuity) points:** {lace_parts['A_points']}")
                    st.write(f"**C (Comorbidity) points:** {lace_parts['C_points']}")
                    st.write(f"**E (ED visits) points:** {lace_parts['E_points']}")

                # Calibration details
                st.markdown("###  Clinical Calibration Process")
                st.markdown(f"""
                <div class="calibration-box">
                    <h4> Evidence-Based Clinical Calibration:</h4>
                    <p><strong>1. Raw ML Prediction:</strong> {raw_ml_prediction:.1%}</p>
                    <p><strong>2. Base Clinical Risk (Age {age}):</strong> {calibration_details['base_clinical']:.1%}</p>
                    <p><strong>3. Condition Factor:</strong> ×{calibration_details['condition_multiplier']:.2f}</p>
                    <p><strong>4. Admission Factor:</strong> ×{calibration_details['admission_multiplier']:.2f}</p>
                    <p><strong>5. Complexity Factor:</strong> ×{calibration_details['complexity_multiplier']:.2f}</p>
                    <p><strong>6. Expected Clinical Risk:</strong> {calibration_details['expected_clinical']:.1%}</p>
                    <p><strong>7. ML Intelligence Factor:</strong> ×{calibration_details['ml_factor']:.2f}</p>
                    <p><strong>8. Final Calibrated Risk:</strong> <strong>{final_prediction:.1%}</strong></p>
                    <p><em> Combines clinical evidence with ML intelligence!</em></p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=final_prediction * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Clinical Risk Score"},
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "yellow"},
                            {'range': [20, 50], 'color': "lightcoral"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Recommendation
                st.markdown("###  Clinical Recommendation")
                if final_prediction < 0.10:
                    recommendation = "**Low Risk**: Standard discharge planning. Routine follow-up in 1–2 weeks."
                elif final_prediction < 0.20:
                    recommendation = "**Moderate Risk**: Enhanced discharge plan. Follow-up within 3–7 days."
                else:
                    recommendation = "**High Risk**: Intensive transitional care. Follow-up within 24–72 hours; consider home health."
                st.info(recommendation)

                # Factors
                st.markdown("###  Risk Factor Analysis")
                factors = []
                if age >= 70:
                    factors.append(f"**Advanced Age ({age})**: Major risk factor")
                elif age < 40:
                    factors.append(f"**Young Age ({age})**: Protective factor")
                if len(conditions) > 0:
                    factors.append(f"**Medical Conditions**: {', '.join(conditions)}")
                else:
                    factors.append("**No Major Comorbidities**: Reduces risk")
                if admission_type == "Emergency":
                    factors.append("**Emergency Admission**: Increases risk by ~30%")
                elif admission_type == "Elective":
                    factors.append("**Elective Admission**: Lower baseline risk")
                for f in factors:
                    st.markdown(f"• {f}")

            except Exception as e:
                st.error(f" Prediction Error: {str(e)}")
        else:
            st.error(" ML Model not available")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Clinical Hospital Readmission Predictor</strong> | Evidence-Based Calibration |
    MSc Data Science Project - University of Greenwich</p>
</div>
""", unsafe_allow_html=True)
