📊 Predicción de Churn en Telecomunicaciones

Análisis, modelo predictivo y recomendaciones de negocio

✅ 1. Introducción

Este proyecto analiza el fenómeno de churn (abandono de clientes) en una empresa de telecomunicaciones.
Se busca identificar qué factores influyen en la pérdida de clientes y construir un modelo capaz de anticipar quiénes tienen mayor riesgo de irse.

El churn es uno de los mayores problemas del sector y tiene impacto directo en ingresos y estabilidad del negocio.

✅ 2. Dataset

Dataset: Telco Customer Churn
Filas: 7032
Columnas: 21

Incluye información sobre:

Datos demográficos

Servicios contratados

Método de pago

Tipo de contrato

Cargo mensual y total

Columna objetivo: Churn (Yes/No)

✅ 3. Objetivos del Proyecto

Analizar los patrones que explican el churn.

Construir un modelo predictivo interpretable.

Identificar segmentos de clientes con mayor riesgo.

Proponer acciones de retención basadas en datos.

✅ 4. Metodología
Preprocesamiento

Conversión de TotalCharges a numérico

Eliminación de 11 valores nulos

One-Hot Encoding para variables categóricas

Escalado para variables numéricas

Modelado

Train/Test Split (80/20)

Modelo final: Regresión Logística

Métricas evaluadas: Accuracy, Recall, Precision, ROC-AUC

Visualización

Gráficos de churn por contrato, tenure, método de pago y cargos mensuales

Importancia de variables

SHAP values para interpretación profunda

✅ 5. Resultados del Análisis (EDA)
🔹 Tasa general de churn: 26.6%

Aproximadamente 1 de cada 4 clientes abandona.

🔹 Tipo de contrato
Contrato	Churn
Month-to-month	42.7%
One year	11.3%
Two year	2.8%

El contrato mensual es el factor más crítico.

🔹 Tenure (meses como cliente)

Clientes que NO hacen churn: 38 meses (mediana)

Clientes que SÍ hacen churn: 10 meses (mediana)

El churn se concentra en clientes nuevos.

🔹 Cargos mensuales

Churn: 74.44 USD

No churn: 61.30 USD

Los clientes que pagan más tienden a abandonar más.

🔹 Método de pago
Método	Churn
Electronic check	45.3%
Bank transfer automatic	16.7%
Credit card automatic	15.3%
Mailed check	19.2%

El método Electronic Check es de alto riesgo.

✅ 6. Modelo Predictivo
Modelo: Regresión Logística

Resultados:

Accuracy: 80%

Recall churn: 57%

ROC-AUC: 0.836

La regresión logística supera a Random Forest en este dataset y ofrece excelente interpretabilidad.

Importancia de variables principales

Variables que aumentan el churn:

Contract: Month-to-month

PaymentMethod: Electronic check

MonthlyCharges altos

Tenure bajo

Falta de TechSupport

Falta de OnlineSecurity

Variables que reducen churn:

Contract: Two year

Tenure alto

Servicios de soporte y seguridad

✅ 7. Recomendaciones de Negocio
1. Migrar clientes de contratos mensuales a contratos anuales

Beneficios, descuentos y campañas dedicadas.

2. Intervenir en los primeros meses del cliente

El churn es más alto entre los meses 1 y 10.

3. Revisión de precios

Clientes con cargos altos son más propensos a abandonar.

4. Migración del método de pago Electronic Check

Ofrecer facilidades para pasarse a métodos automáticos.

5. Fortalecer servicios que reducen churn

TechSupport y OnlineSecurity muestran impacto positivo.

✅ 8. Conclusión

Este proyecto demuestra cómo un análisis completo y un modelo interpretable pueden ofrecer insights valiosos y acciones aplicables para reducir el churn.
La regresión logística alcanzó un rendimiento sólido y permitió identificar los principales factores que impulsan la pérdida de clientes.
