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
    "sex":            pa.Column(str, nullable=True),
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


def estandarizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza todas las columnas de texto:
    - Elimina espacios al inicio y al final
    - Elimina puntos al final del valor
    - Convierte todo a minusculas y luego title case
    Ejemplo: 'OFICIAL', 'oficial', 'Oficial' -> 'Oficial'
    """
    print("\n" + "="*60)
    print("0. ESTANDARIZACION Y NORMALIZACION DE TEXTO")
    print("="*60)

    df_clean      = df.copy()
    cat_cols      = df_clean.select_dtypes(include="object").columns
    cambios_total = 0

    for col in cat_cols:
        antes = df_clean[col].copy()
        df_clean[col] = (df_clean[col]
                         .str.strip()      # Elimina espacios
                         .str.rstrip(".") # Elimina punto al final
                         .str.lower()     # Todo minusculas
                         .str.title())    # Primera letra mayuscula
        cambios = (antes != df_clean[col]).sum()
        if cambios > 0:
            print(f"  {col:<20} → {cambios} valores estandarizados")
            cambios_total += cambios

    if cambios_total == 0:
        print("  ✅ Texto ya estaba estandarizado.")
    else:
        print(f"\n  ✅ Total valores estandarizados: {cambios_total}")

    return df_clean


def estandarizar_target(y: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza el target:
    - Elimina espacios y puntos al final
    Ejemplo: '<=50K.' -> '<=50K', '>50K.' -> '>50K'
    """
    y_clean = y.copy()
    col = y_clean.columns[0]
    y_clean[col] = (y_clean[col]
                    .str.strip()
                    .str.rstrip("."))
    return y_clean


def validar_schema(X: pd.DataFrame) -> bool:
    """1. SCHEMA — tipos, nombres y rangos de columnas."""
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


def validar_nulos(X: pd.DataFrame) -> pd.DataFrame:
    """2. NULOS — porcentaje de valores faltantes por variable."""
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


def validar_distribucion(X: pd.DataFrame) -> None:
    """3. DISTRIBUCION — estadisticos descriptivos y outliers IQR."""
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


def validar_integridad(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    """4. INTEGRIDAD — duplicados y consistencia features/target."""
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


def imputar_datos(X: pd.DataFrame) -> pd.DataFrame:
    """
    5. IMPUTACION — estrategia por tipo de variable.
    Categoricas nominales: imputar con Unknown.
    Razon: preserva la informacion de que el dato faltaba.
    """
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
    """Pipeline completo de validacion."""
    logger.info("Iniciando validacion completa de datos...")
    path = Path(input_dir)
    X = pd.read_parquet(path / 'features.parquet')
    y = pd.read_parquet(path / 'targets.parquet')
    logger.info(f"Datos cargados: {X.shape[0]} filas, {X.shape[1]} columnas")

    # 0. Estandarizar texto ANTES de cualquier validacion
    X = estandarizar_texto(X)
    y = estandarizar_target(y)

    # Mostrar distribucion del target tras estandarizacion
    print("\n--- Distribucion del target tras estandarizacion ---")
    print(y.iloc[:, 0].value_counts())

    # 1. Schema
    schema_ok = validar_schema(X)

    # 2. Nulos
    validar_nulos(X)

    # 3. Distribucion
    validar_distribucion(X)

    # 4. Integridad
    integridad = validar_integridad(X, y)

    # 5. Imputacion
    X_imp = imputar_datos(X)

    # Guardar datos limpios en data/interim/
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    X_imp.to_parquet("data/interim/features_imputadas.parquet", index=False)
    y.to_parquet("data/interim/targets_clean.parquet", index=False)
    logger.info("Guardado: data/interim/features_imputadas.parquet")
    logger.info("Guardado: data/interim/targets_clean.parquet")

    # Artefacto: validation_report.json
    report = {
        "n_rows":       int(X.shape[0]),
        "schema_valid": schema_ok,
        "duplicados":   integridad["duplicados"],
        "consistente":  integridad["consistente"],
        "missing":      X.isnull().sum().to_dict(),
        "missing_pct":  (X.isnull().mean()*100).round(2).to_dict(),
        "target_dist":  y.iloc[:, 0].value_counts().to_dict(),
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
    print(f"Filas:        {result['n_rows']}")
    print(f"Schema OK:    {result['schema_valid']}")
    print(f"Duplicados:   {result['duplicados']}")
    print(f"Target dist:  {result['target_dist']}")
