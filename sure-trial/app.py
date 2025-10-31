import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler from the app directory (relative paths) so the app works on Streamlit Cloud
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'cirrhosis_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

try:
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Model or scaler file not found. Please add 'cirrhosis_model.pkl' and 'scaler.pkl' to the app folder and push to your repo."
        )
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    # Provide helpful debugging info when possible
    try:
        # If model partially loaded, show expected features if available
        if 'model' in locals() and hasattr(model, 'feature_names_in_'):
            st.write("Model expects features:", list(model.feature_names_in_))
    except Exception:
        pass
    st.stop()

# If the model exposes feature names (sklearn >=1.0), prefer those so the app
# automatically adapts to the trained model's expected columns when possible.
if hasattr(model, 'feature_names_in_'):
    EXPECTED_FEATURES = list(model.feature_names_in_)
else:
    # Fallback to the previous hard-coded list if the model doesn't provide feature names
    EXPECTED_FEATURES = ['N_Days', 'Age', 'Ascites', 'Hepatomegaly', 'Spiders',
                         'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                         'Tryglicerides', 'Platelets', 'Prothrombin']

# Pick numerical columns as the intersection of EXPECTED_FEATURES and a known
# set of numerical feature candidates. This reduces breakage if the trained
# model used a different feature ordering or subset.
NUMERICAL_CANDIDATES = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin',
                        'Copper', 'Tryglicerides', 'Platelets', 'Prothrombin',
                        'Alk_Phos', 'SGOT']
NUMERICAL_COLS = [c for c in EXPECTED_FEATURES if c in NUMERICAL_CANDIDATES]

EXPECTED_FEATURES = ['N_Days', 'Age', 'Ascites', 'Hepatomegaly', 'Spiders',
                     'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                     'Tryglicerides', 'Platelets', 'Prothrombin']

NUMERICAL_COLS = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin',
                  'Copper', 'Tryglicerides', 'Platelets', 'Prothrombin']

# Biomarker thresholds for NO Cirrhosis
NO_CIRRHOSIS_RULES = {
    'Bilirubin': 1.2,
    'Albumin': 3.8,
    'Prothrombin': 11.0,
    'Platelets': 190,
}

# Biomarker ranges (min and max) for staging cirrhosis
STAGE_2_RULES = {
    'Bilirubin': (1.2, 2.5),
    'Albumin': (3.0, 3.8),
    'Prothrombin': (11.0, 13.0),
    'Platelets': (150, 190),
}

STAGE_3_RULES = {
    'Bilirubin': (2.5, 3.5),
    'Albumin': (2.5, 3.0),
    'Prothrombin': (13.0, 16.0),
    'Platelets': (100, 150),
}

STAGE_4_RULES = {
    'Bilirubin': (3.5, 100.0),
    'Albumin': (0.0, 2.5),
    'Prothrombin': (16.0, 100.0),
    'Platelets': (0, 100),
}

def check_no_cirrhosis(inp):
    return all([
        inp['Bilirubin'] < NO_CIRRHOSIS_RULES['Bilirubin'],
        inp['Albumin'] > NO_CIRRHOSIS_RULES['Albumin'],
        inp['Prothrombin'] < NO_CIRRHOSIS_RULES['Prothrombin'],
        inp['Platelets'] > NO_CIRRHOSIS_RULES['Platelets']
    ])

def check_stage(inp, stage_rules):
    in_ranges = []
    for biomarker, (min_val, max_val) in stage_rules.items():
        in_range = min_val <= inp[biomarker] <= max_val
        in_ranges.append(in_range)
    # Consider stage matched if majority (>=3 of 4) biomarkers are within the stage range
    return sum(in_ranges) >= 3

def preprocess(inputs):
    proc = {}
    proc['Age'] = inputs['Age'] / 365
    for c in NUMERICAL_COLS:
        if c in inputs and c != 'Age':
            proc[c] = inputs[c]
    proc['Ascites'] = 1 if inputs['Ascites'] == 'Yes' else 0
    proc['Hepatomegaly'] = 1 if inputs['Hepatomegaly'] == 'Yes' else 0
    proc['Spiders'] = 1 if inputs['Spiders'] == 'Yes' else 0
    df = pd.DataFrame(0, index=[0], columns=EXPECTED_FEATURES)
    for c in EXPECTED_FEATURES:
        if c in proc:
            df[c] = proc[c]
    for c in NUMERICAL_COLS:
        if c in df.columns:
            df[c] = np.log(df[c] + 1e-6)
    return df

st.title("Liver Cirrhosis Stage Prediction")
st.header("Input Patient Details")

user_inputs = {}
num_configs = {
    'N_Days': [0, 5000, 1000],
    'Age': [0, 30000, 18000],
    'Bilirubin': [0.0, 50.0, 1.0],
    'Cholesterol': [0.0, 2000.0, 200.0],
    'Albumin': [0.0, 10.0, 3.0],
    'Copper': [0.0, 600.0, 50.0],
    'Tryglicerides': [0.0, 600.0, 100.0],
    'Platelets': [0.0, 800.0, 200.0],
    'Prothrombin': [0.0, 20.0, 10.0],
}
for k, v in num_configs.items():
    user_inputs[k] = st.number_input(k, min_value=v[0], max_value=v[1], value=v[2], format="%.2f" if isinstance(v[2], float) else '%d')
cat_configs = {
    'Ascites': ['No', 'Yes'],
    'Hepatomegaly': ['No', 'Yes'],
    'Spiders': ['No', 'Yes'],
}
for k, opts in cat_configs.items():
    user_inputs[k] = st.selectbox(k, opts)

if st.button("Predict"):
    if check_no_cirrhosis(user_inputs):
        st.success("✅ No Cirrhosis detected based on biomarker thresholds.")
    else:
        # Check stage explicitly using biomarker rules
        if check_stage(user_inputs, STAGE_2_RULES):
            st.success("✅ Cirrhosis detected: Stage 2 (Moderate Liver Damage)")
        elif check_stage(user_inputs, STAGE_3_RULES):
            st.success("✅ Cirrhosis detected: Stage 3 (Severe Liver Damage)")
        elif check_stage(user_inputs, STAGE_4_RULES):
            st.success("✅ Cirrhosis detected: Stage 4 (Advanced Liver Damage)")
        else:
            # If ambiguous, use model prediction for stage
            X = preprocess(user_inputs)
            X_scaled = scaler.transform(X[NUMERICAL_COLS])
            X[NUMERICAL_COLS] = X_scaled
            probs = model.predict_proba(X)[0]
            max_prob = max(probs)
            pred_stage = model.classes_[np.argmax(probs)]
            if max_prob < 0.5:
                st.info("ℹ️ Cirrhosis stage uncertain. No clear match from biomarkers or model with high confidence.")
            else:
                st.success(f"✅ Cirrhosis predicted by model: Stage {int(pred_stage)} (Confidence {max_prob:.2f})")
