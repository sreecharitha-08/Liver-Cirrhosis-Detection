# 🩺 Liver Cirrhosis Detection using Machine Learning

## 📌 Overview

This project focuses on predicting the **stage of liver cirrhosis (1–4)** using machine learning techniques based on patient clinical data. Early prediction of disease severity can support better medical decision-making and improve patient outcomes.

---

## 🎯 Objectives

* Predict cirrhosis stage using patient health indicators
* Compare multiple machine learning models
* Identify the most influential clinical features

---

## 🧠 Models Implemented

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest

---

## ⚙️ Workflow

### 🔹 Data Preprocessing

* Handled missing values
* Encoded categorical features (Sex, Ascites, etc.)
* Feature scaling using standardization

### 🔹 Exploratory Data Analysis (EDA)

* Distribution analysis of clinical features
* Correlation analysis between variables
* Visualization using heatmaps and plots

### 🔹 Feature Engineering

* Selected relevant features impacting disease stage
* Converted categorical data into numerical format

### 🔹 Model Training

* Split dataset into training and testing sets
* Trained multiple models for comparison

### 🔹 Model Evaluation

* Accuracy score
* Confusion matrix
* Performance comparison across models

---

## 📊 Results

* Random Forest achieved the highest accuracy among all models
* Key factors influencing cirrhosis stage include:

  * Bilirubin levels
  * Albumin levels
  * Age and Prothrombin time

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## 🚀 Future Improvements

* Hyperparameter tuning for better accuracy
* Integration with a web application (Flask/Streamlit)
* Deployment for real-time prediction

---

## 📂 Dataset

* Public dataset from Kaggle (Cirrhosis Prediction Dataset)

---

## 💡 Conclusion

This project demonstrates how machine learning can be effectively applied in healthcare to predict disease progression and assist in early diagnosis.


