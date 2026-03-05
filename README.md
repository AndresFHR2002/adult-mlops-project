# Predicción de Ingresos con el Dataset Adult (UCI ML Repository)

## 1. Introducción

Este proyecto implementa un pipeline completo de **MLOps (Machine Learning Operations)** aplicado al
dataset **Adult del UCI Machine Learning Repository**. El objetivo central es predecir si el ingreso
anual de una persona supera los **$50.000 USD**, utilizando variables demográficas y laborales
provenientes del censo de Estados Unidos de 1994.

Más allá del problema de clasificación en sí, el proyecto demuestra cómo llevar un modelo desde un
notebook exploratorio hasta un **pipeline reproducible, versionado y trazable**, aplicando las mejores
prácticas de la industria: validación de datos con esquema, imputación controlada, feature engineering
serializado, experiment tracking con MLflow, y orquestación del pipeline con DVC.


---

## 2. Objetivos

### Objetivo General
Construir un pipeline de MLOps de nivel 1 que automatice el ciclo de vida completo de un modelo de
clasificación binaria, desde la ingesta de datos hasta la evaluación con artefactos visuales.

### Objetivos Específicos

- **Ingesta automatizada:** descargar el dataset Adult directamente desde UCI usando `ucimlrepo` y
  almacenarlo en formato Parquet para eficiencia.

- **Validación robusta:** verificar el esquema, tipos de datos, rangos, valores nulos, duplicados y
  consistencia entre features y target usando `pandera`.

- **Estandarización y limpieza:** normalizar el texto de las variables categóricas (mayúsculas,
  espacios, puntos) para garantizar consistencia en los datos.

- **Imputación informativa:** tratar los valores faltantes en variables categóricas con la categoría
  `"Unknown"`, preservando la información de que el dato estaba ausente.

- **Feature engineering reproducible:** aplicar transformaciones estadísticas (StandardScaler,
  OrdinalEncoder, Target Encoding, Log Transform) y serializar cada transformador como artefacto
  `.joblib` para garantizar que el preprocesamiento en producción sea idéntico al de entrenamiento.

- **Entrenamiento con experiment tracking:** entrenar un Gradient Boosting Classifier con manejo de
  desbalance de clases y registrar cada experimento en MLflow con sus parámetros, métricas y modelo.

- **Evaluación con artefactos visuales:** generar automáticamente la matriz de confusión, curva ROC,
  distribución de probabilidades e importancia de variables al finalizar el entrenamiento.

- **Pipeline reproducible:** orquestar las 5 etapas del pipeline con DVC para que cualquier persona
  pueda reproducir el experimento completo con un solo comando (`dvc repro`).

---

## 3. Fuente de Datos

### Adult Dataset — UCI Machine Learning Repository

| Atributo | Descripción |
|---|---|
| **Nombre** | Adult Census Income |
| **Fuente** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) |
| **ID en UCI** | 2 |
| **Año** | 1994 (Censo de EE.UU.) |
| **Autores** | Ronny Kohavi, Barry Becker |
| **Registros totales** | 48.842 |
| **Variables** | 14 (6 numéricas / 8 categóricas) |
| **Variable respuesta** | `income` — binaria: `<=50K` / `>50K` |
| **Formato almacenado** | Parquet (`.parquet`) vía `ucimlrepo` |
| **Acceso en Python** | `from ucimlrepo import fetch_ucirepo` → `fetch_ucirepo(id=2)` |

---

## 4. Información de las Variables

### Variables Numéricas

| Variable | Rango | Descripción | Transformación |
|---|---|---|---|
| `age` | 17 – 90 | Edad de la persona | StandardScaler |
| `fnlwgt` | > 0 | Peso censal (factor de expansión) | **Eliminada** |
| `education-num` | 1 – 16 | Años de educación numéricos | OrdinalEncoder |
| `capital-gain` | ≥ 0 | Ganancia de capital anual | Log1p transform |
| `capital-loss` | ≥ 0 | Pérdida de capital anual | Log1p transform |
| `hours-per-week` | 1 – 99 | Horas trabajadas por semana | StandardScaler |

### Variables Categóricas

| Variable | Categorías principales | Nulos | Transformación |
|---|---|---|---|
| `workclass` | Private, Self-emp, Gov... | 963 (1.97%) | OrdinalEncoder + imputar "Unknown" |
| `education` | Bachelors, HS-grad, Masters... | 0 | Target Encoding |
| `marital-status` | Married, Never-married... | 0 | OrdinalEncoder |
| `occupation` | Exec-managerial, Prof-specialty... | 966 (1.98%) | OrdinalEncoder + imputar "Unknown" |
| `relationship` | Husband, Not-in-family... | 0 | OrdinalEncoder |
| `race` | White, Black, Asian... | 0 | OrdinalEncoder |
| `sex` | Male, Female | 0 | OrdinalEncoder |
| `native-country` | United-States, Mexico... | 274 (0.56%) | OrdinalEncoder + imputar "Unknown" |

### Variable Respuesta

| Variable | Tipo | Valores | Codificación |
|---|---|---|---|
| `income` | Binaria | `<=50K` / `>50K` | 0 / 1 |


---

## 5. Cómo se Hizo el Proyecto

El proyecto siguió un flujo de trabajo ordenado en 5 fases:

### Fase 1 — Exploración y Análisis (EDA)
Se desarrolló el notebook `Notebooks/eda.ipynb` con un análisis exploratorio completo:
- Carga de datos en 3 etapas del pipeline (raw, interim, processed)
- Análisis del efecto de la estandarización de texto en el target
- Visualización de valores nulos antes y después de la imputación con `missingno`

### Fase 2 — Ingesta y Validación
- `src/ingest.py`: descarga automática desde UCI usando `ucimlrepo` y almacenamiento en Parquet
- `src/validate.py`: validación con `pandera` (schema, tipos, rangos), estandarización de texto
  eliminando espacios y puntos finales con `.str.strip().str.rstrip(".")`, detección de nulos y
  duplicados, e imputación con `"Unknown"` para las 3 variables categóricas con nulos

### Fase 3 — Feature Engineering
- `src/features.py`: transformaciones estadísticas aplicadas a cada variable según su naturaleza
- Se eliminó `fnlwgt` usando `remainder="drop"` en el `ColumnTransformer`
- Cada transformador fue serializado con `joblib.dump()` para garantizar reproducibilidad en producción

### Fase 4 — Modelado y Comparación
Se desarrolló el notebook `Notebooks/modeling.ipynb` comparando 3 modelos con validación cruzada
de 5 folds usando F1-macro como métrica:

| Modelo | F1-macro CV | Notas |
|---|---|---|
| Logistic Regression | 0.7176 ± 0.0027 | Baseline sólido |
| Random Forest | 0.7849 ± 0.0034 | Buena capacidad |
| **Gradient Boosting** | **0.8143 ± 0.0043** | **Mejor modelo ✅** |

El Gradient Boosting fue seleccionado y entrenado en `src/train.py` con manejo de desbalance
mediante `compute_sample_weight` y experiment tracking completo en MLflow.

---

## 6. Metodología

### Pipeline de Datos
```
UCI API → Parquet → Estandarizar texto → Imputar nulos → Transformar → Entrenar → Evaluar
```

### Transformaciones Aplicadas

| Variable | Transformación | Motivo |
|---|---|---|
| `age`, `hours-per-week` | StandardScaler | Escala numérica distinta entre variables |
| `workclass`, `occupation`, etc. | OrdinalEncoder | Variables nominales con muchas categorías |
| `education` | Target Encoding | Alta cardinalidad con señal predictiva fuerte |
| `capital-gain`, `capital-loss` | Log1p transform | Distribución muy sesgada a la derecha |
| `fnlwgt` | Eliminada | Factor censal sin valor predictivo directo |
| Variables categóricas con nulos | Imputar con `"Unknown"` | Ausencia informativa — preserva señal |

### Manejo del Desbalance de Clases

El dataset tiene una proporción 76/24. Se optó por `compute_sample_weight(class_weight="balanced")`
porque preserva todos los datos originales sin síntesis artificial y es compatible directamente con
`GradientBoostingClassifier`. La proporción 3:1 no justifica técnicas más agresivas como SMOTE.

### Selección de Métricas

Se priorizó **F1-macro** sobre accuracy porque el dataset está desbalanceado y el accuracy puede
ser engañoso al ignorar el rendimiento en la clase minoritaria (>50K).

---

## 7. Resultados del Modelo

### Métricas Finales — Gradient Boosting (Test set 20%)

| Métrica | Valor | Interpretación |
|---|---|---|
| **Accuracy** | 0.8389 | 83.9% de predicciones correctas |
| **F1 Macro** | 0.8024 | Excelente balance entre clases |
| **F1 <=50K** | 0.8873 | Alta precisión en clase mayoritaria |
| **F1 >50K** | 0.7175 | Buena detección de clase minoritaria |
| **AUC-ROC** | 0.9289 | Capacidad discriminativa sobresaliente |
| **F1 CV (5-fold)** | 0.8143 | El modelo generaliza bien |

### Artefactos Visuales Generados

| Artefacto | Descripción |
|---|---|
| `confusion_matrix.png` | Matriz de confusión absoluta y porcentual |
| `roc_curve.png` | Curva ROC con AUC y punto óptimo de Youden |
| `feature_importance_plot.png` | Importancia de cada variable del modelo |
| `prob_distribution.png` | Distribución de probabilidades predichas por clase |

---

## 8. Estructura del Repositorio
```
adult-mlops-project/
│
├── data/
│   ├── raw/                             # Datos crudos descargados de UCI
│   │   ├── features.parquet             # Variables originales (48.842 × 14)
│   │   └── targets.parquet              # Target original (con variantes de texto)
│   ├── interim/                         # Datos estandarizados e imputados
│   │   ├── features_imputadas.parquet   # Variables limpias sin nulos
│   │   └── targets_clean.parquet        # Target estandarizado (2 clases)
│   └── processed/                       # Datos listos para modelar
│       ├── features_processed.parquet   # Variables transformadas (48.842 × 12)
│       └── targets_binary.parquet       # Target binario (0 / 1)
│
├── Notebooks/
│   ├── eda.ipynb                        # Análisis exploratorio y descriptivo
│   └── modeling.ipynb                   # Comparación de 3 modelos y selección
│
├── src/
│   ├── __init__.py                      # Hace src/ un módulo Python importable
│   ├── ingest.py                        # Descarga UCI y guarda en data/raw/
│   ├── validate.py                      # Valida schema, estandariza e imputa
│   ├── features.py                      # Feature engineering + serializa .joblib
│   ├── train.py                         # Entrena GBClassifier + MLflow tracking
│   └── evaluate.py                      # Genera métricas y gráficos en artifacts/
│
├── models/
│   └── model.pkl                        # Gradient Boosting serializado
│
├── artifacts/
│   ├── metrics.json                     # Métricas del modelo (accuracy, F1, AUC)
│   ├── feature_importance.json          # Importancia de variables (dict JSON)
│   ├── validation_report.json           # Reporte de validación de datos
│   ├── scaler.joblib                    # StandardScaler (age, hours-per-week)
│   ├── encoder.joblib                   # OrdinalEncoder (7 variables categóricas)
│   ├── target_enc.joblib                # TargetEncoder (education)
│   ├── pipeline.joblib                  # ColumnTransformer completo serializado
│   ├── confusion_matrix.png             # Matriz de confusión
│   ├── roc_curve.png                    # Curva ROC con AUC
│   ├── feature_importance_plot.png      # Importancia de variables (gráfico)
│   └── prob_distribution.png            # Distribución de probabilidades por clase
│
├── tests/
│   ├── test_ingest.py                   # Tests de ingesta de datos
│   ├── test_validate.py                 # Tests de validación y estandarización
│   ├── test_features.py                 # Tests de feature engineering
│   └── test_train.py                    # Tests de entrenamiento
│
├── dvc.yaml                             # Pipeline DVC (5 etapas orquestadas)
├── mlflow.db                            # Base de datos MLflow local (SQLite)
└── README.md
