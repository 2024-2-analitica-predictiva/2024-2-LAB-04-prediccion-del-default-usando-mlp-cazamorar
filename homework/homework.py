# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (precision_score, balanced_accuracy_score, 
                             recall_score, f1_score, confusion_matrix)

# Ruta de los archivos
test_path = "files/input/test_default_of_credit_card_clients.csv"
train_path = "files/input/train_default_of_credit_card_clients.csv"

# Paso 1: Limpieza del dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Renombrar la columna objetivo
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # Eliminar la columna ID
    df.drop(columns=["ID"], inplace=True)

    # Eliminar registros con datos faltantes
    df.dropna(inplace=True)

    # Agrupar niveles superiores de EDUCATION en la categoría "others"
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    return df

# Preprocesar los datasets
train_df = preprocess_data(train_path)
test_df = preprocess_data(test_path)

# Paso 2: División de los datasets
x_train = train_df.drop(columns="default")
y_train = train_df["default"]
x_test = test_df.drop(columns="default")
y_test = test_df["default"]

# Paso 3: Crear el pipeline
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_features = [col for col in x_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", MinMaxScaler(), numeric_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=None)),
    ("feature_selection", SelectKBest(score_func=f_classif)),
    #print(SelectKBest())
    ("classifier", MLPClassifier(random_state=69, max_iter=10000, early_stopping=True))
])

# Paso 4: Optimización de hiperparámetros
param_grid = {
    "pca__n_components": [ 28], #None, 10, 15,
    "feature_selection__k": [15], #range(5, len(x_train.columns) + 1, 5)
    "classifier__hidden_layer_sizes": [ (50, 50, 50, 50, 50), (50, 30, 40, 60)], #(50,), (100,),
    "classifier__activation": ["relu"],
    "classifier__alpha": [ 0.01], #0.0001, 0.001,  0.01
    
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid=param_grid, 
    scoring="balanced_accuracy", 
    cv=10, 
    n_jobs=-1, 
    verbose=1,
    refit=True
)

grid_search.fit(x_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros encontrados:", grid_search.best_params_)

# Paso 5: Guardar el modelo
model_path = "files/models/model.pkl.gz"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with gzip.open(model_path, "wb") as f:
    pickle.dump(grid_search, f)

# Paso 6: Calcular métricas de evaluación
def calculate_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }

def calculate_confusion_matrix(model, x, y, dataset_name):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        "true_1": {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]},
    }

metrics = [
    calculate_metrics(grid_search.best_estimator_, x_train, y_train, "train"),
    calculate_metrics(grid_search.best_estimator_, x_test, y_test, "test"),
    calculate_confusion_matrix(grid_search.best_estimator_, x_train, y_train, "train"),
    calculate_confusion_matrix(grid_search.best_estimator_, x_test, y_test, "test"),
]

with gzip.open("files/models/model.pkl.gz", "rb") as file:
    loaded_model = pickle.load(file)
print(type(loaded_model))

for m in metrics:
    print(m)

# Convertir a tipos JSON-serializables
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")
    
# Guardar métricas
output_path = "files/output/metrics.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for metric in metrics:
        json_line = json.dumps(metric, default=convert_to_serializable)
        f.write(json_line + "\n")

model_path = "files/models/model.pkl.gz"
try:
    with gzip.open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Modelo cargado correctamente:", model)
except Exception as e:
    print("Error al cargar el modelo:", e)

print("Modelo entrenado y métricas guardadas correctamente.")
