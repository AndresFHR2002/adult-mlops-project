import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    model_path:  str = "models/model.pkl",
    data_X_path: str = "data/processed/features_processed.parquet",
    data_y_path: str = "data/processed/targets_binary.parquet",
    output_dir:  str = "artifacts",
) -> dict:

    Path(output_dir).mkdir(exist_ok=True)

    # 1. Cargar modelo y datos
    logger.info("Cargando modelo y datos...")
    clf     = joblib.load(model_path)
    X       = pd.read_parquet(data_X_path)
    y       = pd.read_parquet(data_y_path).iloc[:, 0]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # 2. Calcular métricas
    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro":  round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "f1_class0": round(float(f1_score(y_test, y_pred, pos_label=0)), 4),
        "f1_class1": round(float(f1_score(y_test, y_pred, pos_label=1)), 4),
        "auc_roc":   round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    print("\n===== METRICAS =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])}")

    # 3. Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"], ax=axes[0])
    axes[0].set_title("Matriz de Confusión (Absoluta)", fontweight="bold")
    axes[0].set_xlabel("Predicho")
    axes[0].set_ylabel("Real")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"], ax=axes[1])
    axes[1].set_title("Matriz de Confusión (Porcentual)", fontweight="bold")
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Real")

    plt.suptitle("Gradient Boosting — Matriz de Confusión", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    logger.info("Guardado: confusion_matrix.png")

    # 4. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {metrics['auc_roc']:.4f}")
    plt.plot([0, 1], [0, 1], "gray", linestyle="--")
    plt.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC — Gradient Boosting", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("Guardado: roc_curve.png")

    # 5. Importancia de variables
    importancias = pd.Series(
        clf.feature_importances_, index=X_test.columns
    ).sort_values(ascending=True)

    colores = ["#FF5722" if v > 0.1 else "#2196F3" for v in importancias.values]
    plt.figure(figsize=(10, 7))
    importancias.plot(kind="barh", color=colores, alpha=0.85)
    plt.title("Importancia de Variables — Gradient Boosting", fontweight="bold")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_plot.png", bbox_inches="tight")
    plt.close()
    logger.info("Guardado: feature_importance_plot.png")

    # 6. Distribucion de probabilidades
    plt.figure(figsize=(10, 5))
    plt.hist(y_proba[y_test == 0], bins=50, alpha=0.7,
             color="#2196F3", label="<=50K", density=True)
    plt.hist(y_proba[y_test == 1], bins=50, alpha=0.7,
             color="#FF5722", label=">50K", density=True)
    plt.axvline(x=0.5, color="black", linestyle="--", label="Umbral 0.5")
    plt.xlabel("Probabilidad Predicha P(>50K)")
    plt.ylabel("Densidad")
    plt.title("Distribución de Probabilidades por Clase", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prob_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("Guardado: prob_distribution.png")

    # 7. Guardar métricas
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Guardado: metrics.json")

    print("\n===== ARTEFACTOS GENERADOS =====")
    print("  ✅ confusion_matrix.png")
    print("  ✅ roc_curve.png")
    print("  ✅ feature_importance_plot.png")
    print("  ✅ prob_distribution.png")
    print("  ✅ metrics.json")

    return metrics


if __name__ == "_main_":
    result = evaluate()
    print(f"\n===== RESUMEN FINAL =====")
    for k, v in result.items():
        print(f"  {k}: {v}")