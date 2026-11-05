#!/usr/bin/env python
# coding: utf-8

# 1. Librerías

# In[1]:


import pandas as pd # Para manipulación de datos
import statsmodels.api as sm # Para regresión lineal robusta    
import matplotlib.pyplot as plt # Para visualización de datos
import seaborn as sns   # Para visualización de datos
import numpy as np # Para operaciones numéricas


# 2. Carga de datos

# In[2]:


data = pd.read_csv('Alsea_AD2026.csv', encoding='latin1') # Carga de datos desde un archivo CSV


# 3. Validación de datos

# In[3]:


data.info() # Información general sobre el DataFrame, incluyendo tipos de datos y valores nulos


# 4. Estadistica básica

# In[4]:


data.describe().T


# In[5]:


data.hist(bins=20, figsize=(20,15)) # Histograma de cada variable numérica para visualizar su distribución
plt.show() # Mostrar los histogramas


# 5. División de datos

# In[6]:


from sklearn.model_selection import train_test_split # Para dividir el conjunto de datos en entrenamiento y prueba

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42) # Dividir el conjunto de datos en un 80% para entrenamiento y un 20% para prueba, con una semilla aleatoria para reproducibilidad

print(f"Training set size: {len(train_set)} ({len(train_set)/len(data)*100:.1f}%)") # Imprimir el tamaño del conjunto de entrenamiento y su porcentaje respecto al total
print(f"Test set size: {len(test_set)} ({len(test_set)/len(data)*100:.1f}%)") # Imprimir el tamaño del conjunto de prueba y su porcentaje respecto al total


# 6. Copiamos el set de entrenamiento

# In[7]:


data = train_set.copy()


# 7. Visualización gráfica de variables de interés

# In[8]:


data.plot(kind = "scatter", x = "ESG Score", y = "Return On Capital Employed - Mean")


# 8.  Correlación de variables

# In[9]:


numeric_data = data.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
print(corr_matrix)


# Matriz de correlación

# In[10]:


from pandas.plotting import scatter_matrix # Para crear una matriz de dispersión que muestra las relaciones entre varias variables numéricas
attributes = ["LN Revenue from Business Activities - Total", "CSR Strategy Score", "Waste Total", "CO2 Equivalent Emissions Total"]
scatter_matrix(data[attributes], figsize=(12, 8))


# In[11]:


corr_matrix["Return On Capital Employed - Mean"].sort_values(ascending=False) # Correlación de cada variable con la variable objetivo "Return On Capital Employed - Mean", ordenada de mayor a menor


# ## Preparación de los Datos para los Algoritmos de Machine Learning

# 9. Separación de variable dependiente

# In[12]:


data['Return On Capital Employed - Mean'].isnull().sum()
data = data.dropna(subset=['Return On Capital Employed - Mean'])
data['Return On Capital Employed - Mean'].isnull().sum()


# In[13]:


data = train_set.drop("Return On Capital Employed - Mean", axis=1)
data_labels = train_set["Return On Capital Employed - Mean"].copy()


# 10. Limpieza de datos

# In[14]:


from sklearn.impute import SimpleImputer # Para imputar valores faltantes utilizando la mediana de cada columna

cols_to_drop = [
    "ID", "YEAR", "Policy Waste", "Targets Waste", "e-Waste Reduction",
    "Waste Reduction Initiatives", "Cost of Pollution and Waste",
    "CSR Strategy Score Grade"
]

# 1) Quitar columnas no numéricas / no útiles
data_num = data.drop(columns=cols_to_drop, errors="ignore").copy()

# 2) Convertir columnas object a numérico (errores -> NaN)
for col in data_num.select_dtypes(include=["object"]).columns:
    data_num[col] = pd.to_numeric(data_num[col], errors="coerce")

# 3) Imputación por mediana
imputer = SimpleImputer(strategy="median")
data_num_imputed = pd.DataFrame(
    imputer.fit_transform(data_num),
    columns=data_num.columns,
    index=data_num.index
)


# Ahora podemos usar el “imputer” entrenado para transformar el set de datos de entrenamiento al reemplazar los valores omitidos por la mediana.

# In[15]:


X = imputer.transform(data_num) # Matriz de características (variables independientes) con valores imputados


# In[16]:


data_prepared =pd.DataFrame(X, columns=data_num.columns, index=data_num.index) # DataFrame con los datos preparados (imputados) para el modelado


# 11. Pipelines

# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# # Pipelines en Scikit-learn
# 
# ## ¿Qué es un Pipeline?
# 
# Un **Pipeline** es una herramienta de scikit-learn que encadena múltiples pasos de procesamiento de datos y modelado en una única entidad. Automatiza el flujo de trabajo asegurando que:
# 
# - Las transformaciones se apliquen en el orden correcto
# - Los parámetros se ajusten solo en los datos de entrenamiento
# - Las mismas transformaciones se apliquen consistentemente a nuevos datos
# 
# ## Estructura del Pipeline en tu Notebook
# 
# ```python
# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])
# ```
# 
# ### Pasos del Pipeline:
# 
# | Paso | Función | Propósito |
# |------|---------|----------|
# | **Imputer** | `SimpleImputer(strategy='median')` | Rellena valores faltantes (NaN) con la mediana de cada columna |
# | **Scaler** | `StandardScaler()` | Normaliza las características: (x - media) / desv.estándar |
# 
# ## Flujo de Ejecución
# 
# ```
# Datos originales (con NaN)
#         ↓
# [Imputer] → Rellena valores faltantes
#         ↓
# [Scaler] → Normaliza a media=0, desv.est.=1
#         ↓
# Datos transformados listos para el modelo
# ```
# 
# ## ColumnTransformer: Pipelines Avanzados
# 
# En tu código usas `ColumnTransformer` para aplicar diferentes transformaciones a distintos tipos de variables:
# 
# ```python
# full_pipeline = ColumnTransformer([
#     ("num", num_pipeline, num_attributes)
# ],
# remainder="drop"
# )
# ```
# 
# Esto permite:
# - Aplicar `num_pipeline` solo a variables numéricas
# - Descartar el resto de columnas (`remainder="drop"`)
# 
# ## Ventajas
# 
# ✅ **Evita data leakage**: El `fit()` se realiza solo en entrenamiento  
# ✅ **Código limpio**: Una línea aplica múltiples transformaciones  
# ✅ **Reproducibilidad**: Garantiza el mismo orden de transformaciones  
# ✅ **Productivo**: El pipeline entrenado se puede guardar y reutilizar

# In[18]:


num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
data_num_tr = num_pipeline.fit_transform(data_num)
print(data_num_tr)


# Si tuviéramos que transformar variables categóricas y no categóricas una forma de hacerlo de manera rápida es con la función de Column Transformer

# In[19]:


num_attributes = data_num.columns.tolist()

full_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attributes)],
    remainder="drop"
)

data_prepared = full_pipeline.fit_transform(data_num)


# 12. Selección y Entrenamiento del Modelo

# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lin_reg = LinearRegression()
mask = ~np.isnan(data_labels)
data_prepared = data_prepared[mask]
data_labels = data_labels[mask]
lin_reg.fit(data_prepared, data_labels)


# 13. coeficientes y el intercepto

# In[22]:


print("Coefficients:", lin_reg.coef_) 
print("Intercept:", lin_reg.intercept_)


# In[23]:


import statsmodels.api as sm
data_prepared_with_constant = sm.add_constant(data_prepared) 
lin_reg_sm = sm.OLS(data_labels, data_prepared_with_constant).fit() 
print(lin_reg_sm.summary())


# 14. Nombrar a nuestras variables

# In[24]:


import pandas as pd
import statsmodels.api as sm
nombres_finales = data_num.columns.tolist()
data_df_final = pd.DataFrame(data_prepared_with_constant, columns=['const'] + nombres_finales)
data_labels_clean = pd.Series(data_labels).reset_index(drop=True)
data_df_final = data_df_final.reset_index(drop=True)


# In[25]:


lin_reg_sm = sm.OLS(data_labels_clean, data_df_final).fit()
print(lin_reg_sm.summary())


# 15. Visualizar los datos actuales y las predicciones del modelo

# In[26]:


y_pred = lin_reg_sm.predict(data_df_final) 
plt.figure(figsize=(8, 6))
plt.scatter(data_labels, y_pred, alpha=0.7, color="blue", label="Predicted vs Actual")
plt.plot([data_labels.min(), data_labels.max()], [data_labels.min(), data_labels.max()], color="red", linestyle="--", label="Perfect Fit (y=x)")
plt.title("Actual vs Predicted Values") 
plt.xlabel("Actual Values") 
plt.ylabel("Predicted Values") 
plt.legend()
plt.show()


# 16. Regresión Lineal Robusta

# In[27]:


from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
rlm_model = sm.RLM(data_labels, data_prepared_with_constant, M=HuberT()).fit()
print(rlm_model. summary())


# In[28]:


# Regresión lineal robusta usando las variables ya preparadas con nombres de columnas
rlm_model = sm.RLM(data_labels_clean, data_df_final, M=HuberT()).fit()
print(rlm_model.summary())


# ## Resumen Estadístico del Modelo de Regresión Lineal Robusta (RLM)
# 
# El modelo RLM con la norma de Huber proporciona los siguientes resultados clave:
# 
# ### Métricas de Ajuste del Modelo
# - **R-squared**: Indica qué porcentaje de la variabilidad en el retorno sobre capital empleado es explicado por las variables independientes
# - **Adj. R-squared**: R-squared ajustado por el número de variables, más comparable entre modelos
# - **F-statistic**: Prueba la significancia global del modelo
# 
# ### Coeficientes y Significancia
# Para cada variable se reportan:
# - **Coef**: Cambio esperado en la variable dependiente por unidad de cambio en la variable independiente
# - **Std Err**: Error estándar del coeficiente
# - **t**: Estadístico t para la prueba de significancia
# - **P>|t|**: Valor p (significancia: p < 0.05)
# - **[95% Conf. Int.]**: Intervalo de confianza del 95%
# 
# ### Ventajas de RLM sobre Regresión Lineal Ordinaria
# - **Robustez ante outliers**: La norma de Huber penaliza menos los residuos grandes, reduciendo el impacto de valores extremos
# - **Mayor estabilidad**: En presencia de datos atípicos, proporciona estimaciones más confiables
# 
# ### Diagnóstico
# - **Skewness/Kurtosis**: Evalúan la normalidad de los residuos
# - **Durbin-Watson**: Detecta autocorrelación en residuos
# - **Condition Number**: Indica multicolinealidad entre variables
# 
# Este modelo es recomendable cuando existen outliers o datos anómalos en el conjunto de entrenamiento.

# ## Resumen Estadístico del Modelo de Regresión Lineal
# 
# ### Interpretación General
# El modelo explica la relación entre las variables independientes y el **Return On Capital Employed (ROCE)**. Los coeficientes indican el cambio esperado en ROCE por cada unidad de cambio en la variable, manteniendo las demás constantes.
# 
# ### Variables Estadísticamente Significativas (p < 0.05)
# 
# | Variable | Coeficiente | P-value | Interpretación |
# |----------|-------------|---------|---|
# | **Constante** | 11.81 | 0.000 | Valor base de ROCE sin considerar variables |
# | **LN Revenue from Business Activities - Total** | 15.29 | 0.000 | ⭐ **Más importante**: cada aumento logarítmico en ingresos aumenta ROCE en ~15.29 puntos |
# | **LN Total Assets** | -14.68 | 0.000 | ⭐ **Muy importante**: cada aumento logarítmico en activos reduce ROCE en ~14.68 puntos (relación inversa) |
# | **Waste Recycling Ratio** | -3872.89 | 0.000 | Mayor ratio de reciclaje reduce significativamente ROCE |
# | **Waste Recycled To Total Waste** | 3873.33 | 0.000 | Mayor proporción de residuos reciclados aumenta ROCE |
# | **Non-Hazardous Waste** | 4.45 | 0.003 | Incremento moderado en ROCE por unidad de residuos no peligrosos |
# | **CO2 Equivalent Emissions Total** | 1.08 | 0.049 | Ligero aumento en ROCE con mayores emisiones |
# | **Total Debt Percentage of Total Equity** | -0.92 | 0.048 | Menor impacto: mayor endeudamiento reduce ROCE ligeramente |
# | **Provisions - Long-Term - Total** | -0.96 | 0.036 | Provisiones a largo plazo reducen ligeramente ROCE |
# 
# ### Variables NO Significativas (p > 0.05)
# ESG Score, Waste Total, Hazardous Waste, Waste Recycled Total, e-Waste Reduction Score, CSR Strategy Score, LN Debt - Total, Total Liabilities, Provisions - Total, Inventories - Total
# 
# ### Hallazgos Clave
# - **Drivers principales**: Los ingresos operativos y los activos totales son los factores dominantes en la explicación de ROCE
# - **Paradoja de residuos**: Las variables de reciclaje muestran coeficientes opuestos pero simétricos (multicolinealidad perfecta)
# - **Eficiencia operativa**: Mayor volumen de residuos no peligrosos correlaciona con mejor ROCE
# - **Apalancamiento**: Mayor deuda relativa reduce ligeramente la rentabilidad del capital empleado

# 17. Raíz del Error Cuadrático Medio RMSE

# In[29]:


from sklearn.metrics import mean_squared_error


# In[30]:


data_predictions = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(data_labels, data_predictions) 
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# `np.float64(34.734378777467064)` es el valor de **RMSE** del modelo de regresión lineal (`lin_rmse`).
# 
# - **Qué significa**: en promedio, las predicciones del modelo se desvían alrededor de **34.73 unidades** respecto al valor real de `Return On Capital Employed - Mean`.
# - **Tipo de dato**: `np.float64` indica que es un número decimal de alta precisión de NumPy.
# - **Interpretación práctica**: mientras **más bajo** sea el RMSE, **mejor** ajusta el modelo.  
#     Aquí el error es relativamente alto para una variable como ROCE, por lo que el ajuste puede mejorarse.

# 18. Regresión lineal robusta

# In[31]:


data_predictions = rlm_model.predict(data_prepared_with_constant) 
rlm_mse = mean_squared_error(data_labels, data_predictions) 
rlm_rmse = np.sqrt(rlm_mse)
rlm_rmse


# `np.float64(35.20322415688345)` es el valor de **RMSE** de tu modelo (en este caso, el de la regresión robusta: `rlm_rmse`).
# 
# - **Qué mide**: el error promedio de predicción en las mismas unidades de la variable objetivo (`Return On Capital Employed - Mean`).
# - **Interpretación**: el modelo se equivoca, en promedio, en aproximadamente **35.20** puntos de ROCE.
# - **Tipo de dato**: `np.float64` indica que es un número decimal de alta precisión de NumPy.
# - **Comparación rápida**: frente al RMSE lineal (`34.73`), este valor es **ligeramente mayor**, por lo que en ese conjunto el modelo robusto no mejora el error cuadrático medio.

# In[ ]:


from pathlib import Path
import nbformat
from nbconvert import PythonExporter
from docx import Document
from docx.shared import Pt

notebook_path = Path("mi_notebook.ipynb")
python_path = notebook_path.with_suffix(".py")
word_path = notebook_path.with_name(f"{notebook_path.stem}_codigo.docx")

# Exportar las celdas del notebook a un archivo Python
notebook = nbformat.read(notebook_path, as_version=4)
python_code, _ = PythonExporter().from_notebook_node(notebook)
python_path.write_text(python_code, encoding="utf-8")

# Crear un documento Word con el código exportado
document = Document()
document.add_heading("Código Python del notebook", level=1)

for line in python_code.splitlines():
    paragraph = document.add_paragraph()
    run = paragraph.add_run(line or " ")
    run.font.name = "Consolas"
    run.font.size = Pt(9)

document.save(word_path)

print(f"Archivo Python: {python_path.resolve()}")
print(f"Documento Word: {word_path.resolve()}")

