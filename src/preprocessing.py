"""
preprocessing.py — Pipeline de preprocesamiento de texto para el proyecto RNN
Proyecto Final Deep Learning · Maestría en IA · Pontificia Universidad Javeriana · 2026

Autora: Yibby González
Uso:   from src.preprocessing import build_pipeline

ESTADO: stub documentado — Yibby completa la implementación antes del viernes 8.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Constantes (ajustar tras el EDA de Felipe)
# ──────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 20_000   # Top-N palabras más frecuentes
MAX_LEN     = 200      # Percentil 95 de longitudes (pendiente EDA)
PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
SEED        = 42


# ──────────────────────────────────────────────────────────────────────────────
# Limpieza de texto
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpia un texto individual:
    - Convierte a minúsculas.
    - Preserva tildes y caracteres españoles (ñ, á, é, etc.).
    - Elimina URLs, emails, números y caracteres especiales.
    - Normaliza espacios múltiples.

    Args:
        text: Texto crudo a limpiar.

    Returns:
        Texto limpio como string.
    """
    # TODO: Yibby implementa aquí
    raise NotImplementedError("Yibby: implementar clean_text()")


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizador
# ──────────────────────────────────────────────────────────────────────────────

def build_tokenizer(corpus: list[str], vocab_size: int = VOCAB_SIZE) -> dict:
    """
    Construye un vocabulario palabra → índice a partir del corpus de entrenamiento.
    El vocabulario se limita a las `vocab_size` palabras más frecuentes.
    Reserva índice 0 para <PAD> e índice 1 para <UNK>.

    Args:
        corpus    : Lista de textos ya limpios (solo train, nunca val/test).
        vocab_size: Tamaño máximo del vocabulario.

    Returns:
        word2idx: dict {palabra: índice}.
    """
    # TODO: Yibby implementa aquí
    raise NotImplementedError("Yibby: implementar build_tokenizer()")


def encode_sequences(
    texts: list[str],
    word2idx: dict,
    max_len: int = MAX_LEN,
) -> np.ndarray:
    """
    Tokeniza, encodea y aplica padding/truncado a una lista de textos.

    Args:
        texts   : Lista de textos limpios.
        word2idx: Vocabulario construido con build_tokenizer().
        max_len : Longitud fija de salida (padding o truncado).

    Returns:
        Array numpy de shape (n_textos, max_len) con índices enteros.
    """
    # TODO: Yibby implementa aquí
    raise NotImplementedError("Yibby: implementar encode_sequences()")


# ──────────────────────────────────────────────────────────────────────────────
# Splits estratificados
# ──────────────────────────────────────────────────────────────────────────────

def build_splits(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    seed: int = SEED,
    output_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera splits estratificados 70/15/15 y los guarda como CSV.
    SEMILLA FIJA = 42. No usar otra.

    La estratificación garantiza que cada split respete la distribución
    original de clases del dataset (crítico dado el desbalance del dataset:
    la clase 5★ tiene ~9000 muestras vs ~1000 de la clase 2★).
    Esto asegura que train, val y test tengan la misma proporción de clases
    y que las métricas de val/test sean representativas.

    Args:
        df        : DataFrame completo con texto y etiquetas.
        text_col  : Nombre de la columna de texto.
        target_col: Nombre de la columna de etiquetas.
        seed      : Semilla aleatoria (default 42).
        output_dir: Carpeta donde guardar train.csv, val.csv, test.csv.

    Returns:
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validar que las columnas existen
    for col in [text_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

    df = df[[text_col, target_col]].dropna().copy()

    # Split 70 / 30  →  luego el 30 se parte en 15 / 15
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[target_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[target_col],
    )

    # Guardar
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(  output_dir / "val.csv",   index=False)
    test_df.to_csv( output_dir / "test.csv",  index=False)

    # Resumen
    total = len(df)
    print(f"Splits guardados en '{output_dir}/'")
    print(f"  train.csv : {len(train_df):>5} muestras  ({100*len(train_df)/total:.1f}%)")
    print(f"  val.csv   : {len(val_df):>5} muestras  ({100*len(val_df)/total:.1f}%)")
    print(f"  test.csv  : {len(test_df):>5} muestras  ({100*len(test_df)/total:.1f}%)")
    print("\nDistribución de clases en train (debe reflejar el dataset original):")
    dist = train_df[target_col].value_counts(normalize=True).sort_index()
    for cls, pct in dist.items():
        print(f"  Clase {cls}: {pct:.1%}")

    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y_train: np.ndarray | list) -> torch.Tensor:
    """
    Calcula class weights inversamente proporcionales a la frecuencia de clase,
    usando sklearn.utils.class_weight.compute_class_weight con mode='balanced'.

    Args:
        y_train: Array con las etiquetas del set de entrenamiento (0-indexed).

    Returns:
        Tensor 1-D de pesos, uno por clase, listo para pasar a CrossEntropyLoss.
    """
    # TODO: Yibby implementa aquí
    raise NotImplementedError("Yibby: implementar compute_class_weights()")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset de PyTorch
# ──────────────────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """
    Dataset de PyTorch para reseñas tokenizadas.

    Args:
        sequences: Array (n, max_len) de índices enteros.
        labels   : Array (n,) de etiquetas enteras 0-indexed.
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Función orquestadora
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline(config: dict) -> dict:
    """
    Función orquestadora: carga los CSVs, aplica limpieza, tokenización,
    padding y devuelve los DataLoaders listos para entrenar.

    Args:
        config: Diccionario con las claves:
                  data_dir    (str) — carpeta con train/val/test.csv
                  text_col    (str) — columna de texto
                  label_col   (str) — columna de etiquetas
                  vocab_size  (int)
                  max_len     (int)
                  batch_size  (int)

    Returns:
        {
          "train_loader"  : DataLoader,
          "val_loader"    : DataLoader,
          "test_loader"   : DataLoader,
          "word2idx"      : dict,
          "class_weights" : Tensor,
          "vocab_size"    : int,
          "max_len"       : int,
        }
    """
    # TODO: Yibby implementa aquí
    raise NotImplementedError("Yibby: implementar build_pipeline()")


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada: python src/preprocessing.py
# Genera train.csv, val.csv y test.csv en data/
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import torch

    # Semilla global (contrato 4.3)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Rutas
    ROOT     = Path(__file__).resolve().parent.parent  # raíz del proyecto
    DATA_DIR = ROOT / "data"
    CSV_PATH = DATA_DIR / "Big_AHR.csv"

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en '{CSV_PATH}'.\n"
            "Coloca Big_AHR.csv dentro de la carpeta data/ antes de continuar."
        )

    print(f"Cargando dataset desde: {CSV_PATH}")
    df_raw = pd.read_csv(CSV_PATH)
    print(f"Shape original: {df_raw.shape}")
    print(f"Columnas      : {list(df_raw.columns)}\n")

    # Preparar columnas: texto limpio + label 0-4 (PyTorch necesita índices desde 0)
    df_raw["review_text"] = df_raw["review_text"].astype(str).str.strip()
    df_raw["label"]       = df_raw["rating"] - 1   # rating 1-5 → label 0-4

    # Generar y guardar los splits
    build_splits(
        df        = df_raw,
        text_col  = "review_text",
        target_col= "label",
        seed      = SEED,
        output_dir= DATA_DIR,
    )
