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

    Args:
        df        : DataFrame completo con texto y etiquetas.
        text_col  : Nombre de la columna de texto.
        target_col: Nombre de la columna de etiquetas.
        seed      : Semilla aleatoria (default 42).
        output_dir: Carpeta donde guardar train.csv, val.csv, test.csv.

    Returns:
        (train_df, val_df, test_df)
    """
    # TODO: Yibby implementa aquí
    raise NotImplementedError("Yibby: implementar build_splits()")


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
