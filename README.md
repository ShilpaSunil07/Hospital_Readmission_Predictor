# Reducting Patient Readmission Risk: A Predictive Model using Machine Learning

This project was conducted as part of my MSc Data Science dissertation at the University of Greenwich.

The aim is to predict whether a patient is readmitted to hospital within 30 days of discharge.

A Random Forest model was selected as the final model with an AUC score of **0.905**.

A minimal Streamlit interface is used such that patient information can be entered and the predicted readmission risk can be viewed interactively.

## Project Overview 

The project is based on a realistic yet synthesized dataset of hospital admissions.

The data was built to simulate real patients' records without containing any individual or sensitive data.

Steps took place:

- Dataset preparation and creation

- Feature engineering (i.e. comorbidity count, ICU stay, age bins, complexity score)

- Testing and training the model

- Selection of the Random Forest model as the best performer

- Developing a Streamlit application for interactive prediction

## File Structure 

### Main Files

- `models/create_high_accuracy_model.py` – trains the final Random Forest model and saves it as a pipeline.
- `app/main5.py` – Streamlit interface for making predictions.  
- `data/demo/realistic_data_generator.py` – used for creation of synthetic hospital data utilized for this project(MIMIC-III schema derived ) .  
- `realistic_hospital_data.csv` – synthetic hospital data utilized for this project. 
- `requirements.txt` – dependencies for Windows.  
- `requirements_macos_fixed.txt` – dependencies for macOS.

Other files in the project (such as  `src/models/baseline_models.py`, `src/models/auc_optimizer.py`, etc.) were used for experimenting with different models, testing variations, and analysing results. They are not required for running the final version of the project.

### How to run the project windows

1. Create a virtual environment  
   
   python -m venv venv
   venv\Scripts\activate       # Windows

2. Install the required packages
    
    pip install -r requirements.txt          # Windows

3. Run the streamlit interface
    
   streamlit run app/main5.py

### How to run the project mac 
1. Create a virtual environment  

python3 -m venv .venv
source .venv/bin/activate

2. Install the required packages

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements_macos_fixed.txt

3. Run the streamlit interface

python -m streamlit run app/main5.py



## Project Information

**Author :** **Shilpa Sunil**

**MSc Data Science  project, University of Greenwich**

**Supervisor:** **Mr. Chan Sorayudh Chanthan**

**Tools:** Scikit-learn, Pandas, Numpy, Streamlit, Python, Plotly, CSS, HTML

**Python Version:** 3.10 

## Declaration
This coursework is submitted on the understanding that it is my own work and that it has not, in whole or in part, been presented elsewhere for assessment.