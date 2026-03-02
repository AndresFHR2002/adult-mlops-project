import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Columnas por transformacion (fnlwgt se elimina)
NUM_COLS    = ["age", "hours-per-week"]
ORDINAL_COLS = ["workclass", "occupation", "marital-status",
                "relationship", "race", "sex", "native-country"]
TARGET_COLS  = ["education"]
LOG_COLS     = ["capital-gain", "capital-loss"]

def build_preprocessor(y_train=None):
    # num: StandardScaler
    num = Pipeline([
        ("scaler", StandardScaler())
    ])
    # cat: OrdinalEncoder
    cat = Pipeline([
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    ct = ColumnTransformer([
        ("num", num, NUM_COLS),
        ("cat", cat, ORDINAL_COLS),
    ], remainder="drop")  # fnlwgt queda eliminada
    return ct

def run_features(input_dir: str = "data/interim") -> dict:
    logger.info("Iniciando Feature Engineering...")

    # Leer datos imputados (generados por validate.py)
    path = Path(input_dir)
    X = pd.read_parquet(path / "features_imputadas.parquet")
    y = pd.read_parquet("data/interim/targets_clean.parquet")

    # Binarizar target: >50K=1, <=50K=0
    y_binary = (y.iloc[:, 0] == ">50K").astype(int)

    # Log transform a capital-gain y capital-loss
    for col in LOG_COLS:
        X[col] = np.log1p(X[col])
    logger.info("Log-transform aplicado a capital-gain y capital-loss.")

    # Target Encoding para education
    te = TargetEncoder(cols=TARGET_COLS)
    X["education"] = te.fit_transform(X["education"], y_binary)
    logger.info("Target Encoding aplicado a education.")

    # Construir y ajustar preprocesador
    preprocessor = build_preprocessor()
    X_processed  = preprocessor.fit_transform(X)

    # Columnas finales
    feature_names = NUM_COLS + ORDINAL_COLS
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    # Agregar education (ya transformada) y log cols
    X_df["education"]    = X["education"].values
    X_df["capital-gain"] = X["capital-gain"].values
    X_df["capital-loss"] = X["capital-loss"].values

    logger.info(f"Shape tras preprocesamiento: {X_df.shape}")

    # Guardar datos procesados
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_df.to_parquet("data/processed/features_processed.parquet", index=False)
    pd.DataFrame({"income": y_binary}).to_parquet("data/processed/targets_binary.parquet", index=False)
    logger.info("Datos procesados guardados en data/processed/")

    # Serializar artefactos .joblib
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(preprocessor.named_transformers_["num"], "artifacts/scaler.joblib")
    joblib.dump(preprocessor.named_transformers_["cat"], "artifacts/encoder.joblib")
    joblib.dump(te,           "artifacts/target_enc.joblib")
    joblib.dump(preprocessor, "artifacts/pipeline.joblib")
    logger.info("Artefactos guardados: scaler.joblib, encoder.joblib, target_enc.joblib, pipeline.joblib")

    # Resumen
    print("\n===== RESUMEN FEATURE ENGINEERING =====")
    print(f"{'Variable':<25} {'Transformacion':<20} {'Artefacto'}")
    print("-"*65)
    print(f"{'age, hours-per-week':<25} {'StandardScaler':<20} scaler.joblib")
    print(f"{'workclass, occupation':<25} {'OrdinalEncoder':<20} encoder.joblib")
    print(f"{'education':<25} {'Target Encoding':<20} target_enc.joblib")
    print(f"{'capital-gain/loss':<25} {'Log transform':<20} pipeline.joblib")
    print(f"{'fnlwgt':<25} {'Eliminada':<20} —")
    print(f"\nShape final:         {X_df.shape}")
    print(f"Distribucion target: {y_binary.value_counts().to_dict()}")

    return {"shape": X_df.shape, "features": list(X_df.columns)}

if __name__ == "__main__":
    result = run_features()
    print(f"\n✅ Feature Engineering completado.")
    print(f"Features generadas: {result['features']}")

