import argparse
import json
import logging
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import git
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Hiperparametros por defecto
DEFAULT_PARAMS = {
    "n_estimators":  200,
    "learning_rate": 0.1,
    "max_depth":     4,
    "subsample":     0.8,
    "random_state":  42,
}

def get_git_commit() -> str:
    """Obtiene el hash del commit actual de Git para trazabilidad."""
    try:
        repo   = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:7]
    except Exception:
        return "no-git"

def train(X_train, y_train, X_test, y_test, params: dict):
    """
    Entrena el modelo y registra todo en MLflow.
    Estructura exacta de la diapositiva.
    """
    mlflow.set_experiment("adult-income")

    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Tags: dataset version y git commit
        mlflow.set_tags({
            "dataset_version": "Adult UCI ID=2",
            "git_commit":      get_git_commit(),
        })

        # Modelo
        clf = GradientBoostingClassifier(**params)

        # Cross-validation cv=5, scoring=f1_macro
        logger.info("Ejecutando cross-validation (5-fold)...")
        scores = cross_val_score(
            clf, X_train, y_train,
            cv=5, scoring="f1_macro", n_jobs=-1
        )
        logger.info(f"F1 CV: {scores.mean():.4f} +/- {scores.std():.4f}")

        # Entrenamiento final
        clf.fit(X_train, y_train)

        # Metricas en test
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "f1_cv":    round(float(scores.mean()), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "auc_roc":  round(float(roc_auc_score(y_test, y_proba)), 4),
        }

        # Registrar params en MLflow
        mlflow.log_params(params)

        # Registrar metricas en MLflow
        mlflow.log_metric("f1_cv",    metrics["f1_cv"])
        mlflow.log_metric("f1_macro", metrics["f1_macro"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("auc_roc",  metrics["auc_roc"])

        # Registrar modelo en MLflow (model/ + conda.yaml + MLmodel)
        mlflow.sklearn.log_model(clf, "model")
        logger.info("Modelo registrado en MLflow.")

        # Guardar artefactos locales
        Path("models").mkdir(exist_ok=True)
        Path("artifacts").mkdir(exist_ok=True)

        joblib.dump(clf, "models/model.pkl")
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Reporte completo
        print("\n===== ARTEFACTOS REGISTRADOS POR MLFLOW =====")
        print("  model/    → Modelo serializado + conda.yaml + MLmodel")
        print("  metrics/  → F1, accuracy, AUC por cada run")
        print("  params/   → n_estimators, lr, depth")
        print("  tags/     → dataset version, git commit")

        print("\n===== METRICAS =====")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        print(f"\n{classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])}")
        logger.info("Artefactos: models/model.pkl | artifacts/metrics.json")

        return metrics


def parse_args():
    """Permite recibir hiperparametros por consola."""
    parser = argparse.ArgumentParser(description="Entrenamiento Adult Income")
    parser.add_argument("--n_estimators",  type=int,   default=200)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth",     type=int,   default=4)
    parser.add_argument("--subsample",     type=float, default=0.8)
    parser.add_argument("--random_state",  type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    # Leer argumentos de consola
    args   = parse_args()
    params = {
        "n_estimators":  args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth":     args.max_depth,
        "subsample":     args.subsample,
        "random_state":  args.random_state,
    }

    logger.info("Cargando datos procesados...")
    X = pd.read_parquet("data/processed/features_processed.parquet")
    y = pd.read_parquet("data/processed/targets_binary.parquet").iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=params["random_state"], stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    train(X_train, y_train, X_test, y_test, params)
