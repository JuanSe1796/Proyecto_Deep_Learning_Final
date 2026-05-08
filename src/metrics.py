"""
metrics.py — Métricas estandarizadas, JSON y gráficas para el proyecto RNN
Proyecto Final Deep Learning · Maestría en IA · Pontificia Universidad Javeriana · 2026

Autor: Daniel Ruiz
Uso:  from src.metrics import compute_metrics, save_metrics_json, plot_training_curves,
                              plot_confusion_matrix

Flujo típico:
    y_true, y_pred = evaluate(model, test_loader, checkpoint_path, config)
    metrics_dict   = compute_metrics(y_true, y_pred, metrics_dict, class_names)
    save_metrics_json(metrics_dict, "results/lstm_metrics.json")
    plot_training_curves(metrics_dict, save_path="figures/lstm_curves.png")
    plot_confusion_matrix(metrics_dict, class_names, save_path="figures/lstm_cm.png")
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Orden de clases del dataset (1–5 estrellas). Ajustar si el dataset usa otros labels.
DEFAULT_CLASS_NAMES = ["1", "2", "3", "4", "5"]

# ──────────────────────────────────────────────────────────────────────────────
# Cálculo de métricas
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    metrics_dict: dict,
    class_names: list[str] | None = None,
) -> dict:
    """
    Calcula accuracy, precision/recall/F1 macro, F1 por clase y matriz de
    confusión, e inyecta los resultados en metrics_dict["metrics"].

    Args:
        y_true       : Etiquetas reales (enteros 0-indexed).
        y_pred       : Predicciones del modelo (enteros 0-indexed).
        metrics_dict : Diccionario parcial devuelto por train(). Se modifica
                       in-place y también se retorna.
        class_names  : Lista de nombres de clase en el mismo orden que los índices.
                       Default: ["1","2","3","4","5"].

    Returns:
        metrics_dict actualizado con la sección "metrics" completa.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    f1_per_class_arr = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Mapear índice → nombre de clase ("0" → "1", etc.)
    f1_per_class = {
        class_names[i]: round(float(v), 6)
        for i, v in enumerate(f1_per_class_arr)
    }

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics_dict["metrics"] = {
        "accuracy":        round(float(accuracy), 6),
        "precision_macro": round(float(precision), 6),
        "recall_macro":    round(float(recall), 6),
        "f1_macro":        round(float(f1_macro), 6),
        "f1_per_class":    f1_per_class,
        "confusion_matrix": cm,
    }

    # Imprimir resumen en consola
    print("\n── Métricas de evaluación (test set) ──────────────────────")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Precision macro : {precision:.4f}")
    print(f"  Recall macro    : {recall:.4f}")
    print(f"  F1 macro        : {f1_macro:.4f}")
    print("  F1 por clase:")
    for cls, val in f1_per_class.items():
        print(f"    Clase {cls}: {val:.4f}")
    print("────────────────────────────────────────────────────────────\n")

    return metrics_dict


# ──────────────────────────────────────────────────────────────────────────────
# Guardar JSON estandarizado
# ──────────────────────────────────────────────────────────────────────────────

def save_metrics_json(metrics_dict: dict, path: str | Path) -> None:
    """
    Guarda el metrics_dict completo como archivo JSON en la ruta indicada.

    Args:
        metrics_dict: Diccionario en formato estandarizado del equipo.
        path        : Ruta de destino, p. ej. "results/lstm_metrics.json".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    print(f"  JSON guardado → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Gráfica: curvas de loss y accuracy
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    metrics_dict: dict,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Genera la figura con dos subplots: curvas de loss y curvas de accuracy
    (train vs. validación). Guarda como PNG si se indica save_path.

    Args:
        metrics_dict: Diccionario estandarizado con la sección "training".
        save_path   : Ruta PNG de destino. Si es None, no guarda archivo.
        show        : Si True, muestra la figura en pantalla (útil en Colab).
    """
    training = metrics_dict["training"]
    model_name = metrics_dict["model_name"]

    epochs = range(1, len(training["loss_history"]) + 1)
    best_epoch = training["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Curvas de entrenamiento — {model_name}", fontsize=13)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, training["loss_history"],     label="Train loss",  color="#2196F3")
    ax.plot(epochs, training["val_loss_history"], label="Val loss",    color="#F44336", linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.8, label=f"Mejor época ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss (CrossEntropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, training["acc_history"],     label="Train acc",  color="#4CAF50")
    ax.plot(epochs, training["val_acc_history"], label="Val acc",    color="#FF9800", linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.8, label=f"Mejor época ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Curvas guardadas → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Gráfica: matriz de confusión
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    metrics_dict: dict,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    normalize: bool = True,
) -> None:
    """
    Genera la matriz de confusión como heatmap y la guarda como PNG.

    Args:
        metrics_dict: Diccionario estandarizado con metrics["confusion_matrix"].
        class_names : Lista de nombres de clase. Default: ["1","2","3","4","5"].
        save_path   : Ruta PNG de destino.
        show        : Si True, muestra la figura en pantalla.
        normalize   : Si True, normaliza por fila (muestra % de predicciones).
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    cm = np.array(metrics_dict["metrics"]["confusion_matrix"])
    model_name = metrics_dict["model_name"]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
        title_suffix = "(normalizada por fila)"
    else:
        cm_plot = cm
        fmt = "d"
        title_suffix = "(conteos absolutos)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
    )
    ax.set_title(f"Matriz de confusión — {model_name}\n{title_suffix}", fontsize=12)
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Etiqueta predicha")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Matriz de confusión guardada → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Atajo: pipeline completo de post-evaluación
# ──────────────────────────────────────────────────────────────────────────────

def finalize_and_save(
    y_true: list[int],
    y_pred: list[int],
    metrics_dict: dict,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    class_names: list[str] | None = None,
    show_plots: bool = False,
) -> dict:
    """
    Pipeline completo de post-evaluación:
      1. Calcula métricas de clasificación.
      2. Guarda JSON estandarizado.
      3. Genera y guarda curvas de entrenamiento (PNG).
      4. Genera y guarda matriz de confusión (PNG).

    Args:
        y_true, y_pred  : Listas devueltas por evaluate().
        metrics_dict    : Diccionario parcial devuelto por train().
        results_dir     : Carpeta donde guardar el JSON.
        figures_dir     : Carpeta donde guardar los PNG.
        class_names     : Nombres de las clases. Default: ["1","2","3","4","5"].
        show_plots      : Si True, muestra figuras en pantalla (Colab).

    Returns:
        metrics_dict completo y actualizado.
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    model_name  = metrics_dict["model_name"]

    # 1. Métricas
    metrics_dict = compute_metrics(y_true, y_pred, metrics_dict, class_names)

    # 2. JSON
    save_metrics_json(metrics_dict, results_dir / f"{model_name}_metrics.json")

    # 3. Curvas de entrenamiento
    plot_training_curves(
        metrics_dict,
        save_path=figures_dir / f"{model_name}_curves.png",
        show=show_plots,
    )

    # 4. Matriz de confusión
    plot_confusion_matrix(
        metrics_dict,
        class_names=class_names,
        save_path=figures_dir / f"{model_name}_confusion_matrix.png",
        show=show_plots,
    )

    return metrics_dict
