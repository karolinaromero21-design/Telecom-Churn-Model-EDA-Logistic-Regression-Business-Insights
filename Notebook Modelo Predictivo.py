# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:58:40 2025

@author: karol
"""

# Predicción de Churn en Telecomunicaciones

## Introducción
Este notebook analiza el churn de clientes utilizando el dataset **Telco Customer Churn**. Incluye limpieza de datos, análisis exploratorio, modelado con **Regresión Logística**, interpretación con SHAP y conclusiones orientadas al negocio.

# Telecom Churn Prediction - Notebook Estructurado

# 01 - Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
import shap
import os

# 02 - Carga del dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df.head()

# 03 - Limpieza de datos
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# 04 - Exploración de datos (EDA)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Distribución general del churn')
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn por tipo de contrato')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='PaymentMethod', hue='Churn')
plt.title('Churn por método de pago')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title('Distribución de tenure por churn')
plt.show()

# 05 - Guardado de imágenes
os.makedirs("images", exist_ok=True)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Distribución general del churn')
plt.savefig("images/churn_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn por tipo de contrato')
plt.xticks(rotation=45)
plt.savefig("images/churn_by_contract.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='PaymentMethod', hue='Churn')
plt.title('Churn por método de pago')
plt.xticks(rotation=45)
plt.savefig("images/churn_by_payment.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title('Distribución de tenure por churn')
plt.savefig("images/tenure_by_churn.png", dpi=300, bbox_inches='tight')
plt.show()

# 06 - Preparación del modelado
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn', 'customerID'])

cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(include=['int64','float64']).columns

# 07 - Pipeline + Logistic Regression
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

log_reg_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg_model.fit(X_train, y_train)

# 08 - Evaluación del modelo
y_pred = log_reg_model.predict(X_test)
y_proba = log_reg_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión – Regresión Logística")
plt.show()

# 09 - Interpretación: coeficientes
ohe = log_reg_model.named_steps['preprocessor'].named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(cat_cols)

final_features = np.concatenate([ohe_features, num_cols])

coeffs = log_reg_model.named_steps['classifier'].coef_[0]

feature_importance = pd.DataFrame({
    'feature': final_features,
    'coef': coeffs
}).sort_values('coef', ascending=False)

feature_importance.head()

# 10 - SHAP summary plot
explainer = shap.LinearExplainer(
    log_reg_model.named_steps['classifier'],
    log_reg_model.named_steps['preprocessor'].transform(X_train),
    feature_names=final_features
)

shap_values = explainer.shap_values(
    log_reg_model.named_steps['preprocessor'].transform(X_test)
)

shap.summary_plot(shap_values, feature_names=final_features)
