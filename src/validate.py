import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pandera as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

schema = pa.DataFrameSchema({
    "age":            pa.Column(int, checks=pa.Check.in_range(17, 90)),
    "workclass":      pa.Column(str, nullable=True),
    "education-num":  pa.Column(int, checks=pa.Check.in_range(1, 16)),
    "sex":            pa.Column(str, checks=pa.Check.isin(["Male", "Female"])),
    "hours-per-week": pa.Column(int, checks=pa.Check.in_range(1, 99)),
    "capital-gain":   pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
    "capital-loss":   pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
    "occupation":     pa.Column(str, nullable=True),
    "native-country": pa.Column(str, nullable=True),
    "fnlwgt":         pa.Column(int, checks=pa.Check.greater_than(0)),
    "education":      pa.Column(str, nullable=False),
    "marital-status": pa.Column(str, nullable=False),
    "relationship":   pa.Column(str, nullable=False),
    "race":           pa.Column(str, nullable=False),
}, coerce=True, strict=False)

def validar_schema(X):
    print("\n" + "="*60)
    print("1. SCHEMA — Tipos y rangos de columnas")
    print("="*60)
    try:
        schema.validate(X, lazy=True)
        print("✅ Schema validado correctamente.")
        return True
    except pa.errors.SchemaErrors as e:
        print(f"⚠️  Errores: {len(e.failure_cases)}")
        print(e.failure_cases)
        return False

def validar_nulos(X):
    print("\n" + "="*60)
    print("2. NULOS — Valores faltantes por columna")
    print("="*60)
    missing     = X.isnull().sum()
    missing_pct = (X.isnull().mean() * 100).round(2)
    tabla = pd.DataFrame({
        "Columna":      missing.index,
        "Nulos":        missing.values,
        "Porcentaje %": missing_pct.values,
        "Tiene nulos":  ["⚠️  Si" if v > 0 else "✅ No" for v in missing.values],
    })
    print(tabla.to_string(index=False))
    return tabla

def validar_distribucion(X):
    print("\n" + "="*60)
    print("3. DISTRIBUCION — Estadisticos descriptivos")
    print("="*60)
    print(X.describe().round(2))
    print("\n--- Outliers por IQR ---")
    for col in X.select_dtypes(include=[np.number]).columns:
        Q1    = X[col].quantile(0.25)
        Q3    = X[col].quantile(0.75)
        IQR   = Q3 - Q1
        n_out = ((X[col] < Q1 - 1.5*IQR) | (X[col] > Q3 + 1.5*IQR)).sum()
        print(f"  {col:<20} {'⚠️  '+str(n_out)+' outliers' if n_out > 0 else '✅ Sin outliers'}")

def validar_integridad(X, y):
    print("\n" + "="*60)
    print("4. INTEGRIDAD — Duplicados y consistencia")
    print("="*60)
    duplicados  = int(X.duplicated().sum())
    consistente = len(X) == len(y)
    print(f"  Filas duplicadas: {duplicados} ({round(duplicados/len(X)*100,2)}%)")
    print(f"  Total filas X:    {len(X)}")
    print(f"  Total filas y:    {len(y)}")
    print(f"  Consistencia X/y: {'✅ OK' if consistente else '⚠️  ERROR'}")
    return {"duplicados": duplicados, "consistente": consistente}

def imputar_datos(X):
    print("\n" + "="*60)
    print("5. IMPUTACION — Mejor estrategia por variable")
    print("="*60)
    print(f"\n  {'Variable':<20} {'Estrategia':<22} {'Nulos':<8} {'Razon'}")
    print(f"  {'-'*75}")
    X_imp = X.copy()
    for col in ["workclass", "occupation", "native-country"]:
        n = X_imp[col].isnull().sum()
        X_imp[col] = X_imp[col].fillna("Unknown")
        print(f"  {col:<20} {'Unknown (categoria)':<22} {n:<8} Nominal, ausencia informativa")
    nulos = X_imp.isnull().sum().sum()
    print(f"\n  {'✅ Imputacion completa.' if nulos == 0 else '⚠️  Quedan '+str(nulos)+' nulos.'}")
    return X_imp

def validate_features(input_dir: str = 'data/raw') -> dict:
    logger.info("Iniciando validacion completa de datos...")
    path = Path(input_dir)
    X = pd.read_parquet(path / 'features.parquet')
    y = pd.read_parquet(path / 'targets.parquet')
    logger.info(f"Datos cargados: {X.shape[0]} filas, {X.shape[1]} columnas")

    schema_ok  = validar_schema(X)
    validar_nulos(X)
    validar_distribucion(X)
    integridad = validar_integridad(X, y)
    X_imp      = imputar_datos(X)

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    X_imp.to_parquet("data/interim/features_imputadas.parquet", index=False)
    logger.info("Guardado: data/interim/features_imputadas.parquet")

    report = {
        "n_rows":       int(X.shape[0]),
        "schema_valid": schema_ok,
        "duplicados":   integridad["duplicados"],
        "consistente":  integridad["consistente"],
        "missing":      X.isnull().sum().to_dict(),
        "missing_pct":  (X.isnull().mean()*100).round(2).to_dict(),
    }
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Artefacto guardado: artifacts/validation_report.json")
    print("\n✅ Validacion completa.")
    return report

if __name__ == "__main__":
    result = validate_features()
    print(f"\n===== RESUMEN =====")
    print(f"Filas:      {result['n_rows']}")
    print(f"Schema OK:  {result['schema_valid']}")
    print(f"Duplicados: {result['duplicados']}")