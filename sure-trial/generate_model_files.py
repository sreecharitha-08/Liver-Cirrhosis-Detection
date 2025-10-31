import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier 

# --- 1. Load Data ---
try:
    # Use raw string for Windows path to avoid SyntaxWarning
    cir = pd.read_csv(r'D:\datasets\cirrhosis.csv') 
except FileNotFoundError:
    print("Error: 'cirrhosis.csv' not found. Please ensure the file is accessible at 'D:\\datasets\\cirrhosis.csv'.")
    exit() 

# --- 2. Initial Data Cleaning and Feature Engineering ---
# Keep only stages 2, 3, 4 as per previous discussions. Stage 1 is excluded.
cir = cir[cir["Stage"] != 1].copy() 
cir.drop(['ID'], axis=1, inplace=True)
cir["Age"] = cir["Age"] / 365

# --- 3. Handle Missing Values ---
cir.dropna(axis=0, subset=["Stage"], inplace=True)
cir["Stage"] = cir["Stage"].astype(int)

# Identify ALL original numerical columns for log transformation and potential scaling
ALL_ORIGINAL_NUMERICAL_COLS = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                               'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

# Impute numerical missing values with the median and apply log transformation
for c in ALL_ORIGINAL_NUMERICAL_COLS:
    if c in cir.columns:
        cir[c].fillna(cir[c].median(), inplace=True) # Impute before log
        cir[c] = np.log(cir[c] + 1e-6) # Apply log transform

# Impute categorical missing values with the mode
categorical_columns_for_imputation = cir.select_dtypes(include=('object')).columns
for c in categorical_columns_for_imputation:
    cir[c].fillna(cir[c].mode().values[0], inplace=True)

# --- 4. Column Encoding ---
cir['Sex'] = cir['Sex'].replace({'M': 0, 'F': 1})
cir['Ascites'] = cir['Ascites'].replace({'N': 0, 'Y': 1})
cir['Drug'] = cir['Drug'].replace({'D-penicillamine': 1, 'Placebo': 0})
cir['Hepatomegaly'] = cir['Hepatomegaly'].replace({'N': 0, 'Y': 1})
cir['Spiders'] = cir['Spiders'].replace({'N': 0, 'Y': 1})
cir['Edema'] = cir['Edema'].replace({'N': 0, 'Y': 1, 'S': -1})
cir['Status'] = cir['Status'].replace({'C': 0, 'CL': 1, 'D': -1})

X = cir.drop(['Status', 'Stage'], axis=1)
y = cir['Stage']

# --- 5. Split Data and Oversampling (SMOTE) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

sm = SMOTE(k_neighbors=3, random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# --- Apply Feature Selection FIRST on the resampled data ---
best_selector = SelectKBest(score_func=f_classif, k=12) # Selecting top 12 features
best_selector.fit(X_train_resampled, y_train_resampled)

best_feature_names = X_train_resampled.columns[best_selector.get_support()].tolist()

# Now, filter X_train_resampled and X_test to only include the selected features
X_train_final = X_train_resampled[best_feature_names].copy() # .copy() to avoid SettingWithCopyWarning
X_test_final = X_test[best_feature_names].copy() # Ensure X_test also gets only the selected features, .copy()

# --- Determine which of the SELECTED features are numerical (for scaling) ---
NUMERICAL_FEATURES_FOR_SCALING = [
    col for col in best_feature_names if col in ALL_ORIGINAL_NUMERICAL_COLS
]

# --- 6. Feature Scaling (StandardScaler) ---
scaler = StandardScaler()
# FIT THE SCALER ONLY ON THE *SELECTED* NUMERICAL FEATURES
scaler.fit(X_train_final[NUMERICAL_FEATURES_FOR_SCALING])

# Transform both training and testing numerical data
X_train_final[NUMERICAL_FEATURES_FOR_SCALING] = scaler.transform(X_train_final[NUMERICAL_FEATURES_FOR_SCALING])
X_test_final[NUMERICAL_FEATURES_FOR_SCALING] = scaler.transform(X_test_final[NUMERICAL_FEATURES_FOR_SCALING])


# --- 7. Model Training ---
model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="log_loss", max_depth=10, min_samples_leaf=4)
model.fit(X_train_final, y_train_resampled)

# --- 8. Save the trained model and scaler ---
with open('cirrhosis_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file) # Save the numerical-only fitted scaler

print("✅ Model and scaler saved successfully as 'cirrhosis_model.pkl' and 'scaler.pkl'.")
print("These files are now ready to be used by your Streamlit app (`app.py`).")
print("\n--- 🌟🌟🌟 IMPORTANT: Copy this list EXACTLY for 'EXPECTED_FEATURES' in app.py 🌟🌟🌟 ---")
print(best_feature_names) # THIS IS THE LIST YOU NEED TO COPY!
print("\n--- 🌟🌟🌟 ALSO COPY THIS LIST EXACTLY for 'NUMERICAL_COLS_TO_PROCESS' in app.py 🌟🌟🌟 ---")
print(NUMERICAL_FEATURES_FOR_SCALING) # THIS IS THE LIST YOU NEED TO COPY!
