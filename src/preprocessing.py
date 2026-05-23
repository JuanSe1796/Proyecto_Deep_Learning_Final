"""
preprocessing.py — Pipeline de preprocesamiento de texto para el proyecto RNN
Proyecto Final Deep Learning · Maestría en IA · Pontificia Universidad Javeriana · 2026

Autora: Yibby González
Uso:   from src.preprocessing import build_pipeline

Descripción general
-------------------
Este módulo implementa el pipeline completo de preparación de texto:

1. clean_text()            — Limpieza: lowercase, tildes, URLs, emails, números,
                             caracteres especiales, normalización de espacios.
2. build_tokenizer()       — Construye vocabulario palabra→índice desde el corpus
                             de entrenamiento (top-N palabras). <PAD>=0, <UNK>=1.
3. encode_sequences()      — Tokeniza, encodea y aplica padding/truncado.
4. build_splits()          — Splits estratificados 70/15/15 con semilla fija 42.
5. compute_class_weights() — Pesos inversamente proporcionales a la frecuencia.
6. ReviewDataset           — Dataset de PyTorch.
7. build_pipeline()        — Función orquestadora.

Decisiones de diseño documentadas
----------------------------------
- TILDES: Se PRESERVAN. En español, "mas" ≠ "más". Eliminarlas introduce
  ambigüedades léxicas perjudiciales para el análisis de sentimiento.

- STOP WORDS: Se CONSERVAN en el pipeline clásico. Las RNNs aprenden
  relaciones secuenciales y palabras como "no", "nunca", "pero" son
  fundamentales para capturar negaciones en sentimiento. El transformer
  de Sebastián usa su propio tokenizador y no depende de este pipeline.

- TOKENIZACIÓN por palabras (no subwords): Los modelos LSTM/BiLSTM clásicos
  funcionan bien con vocabularios de palabras. Top-20,000 cubre ~95-98% del
  corpus español con una embedding matrix manejable (20k×128 ≈ 10M params).
  Para el transformer, HuggingFace usa su propio tokenizador subword.

- MAX_LEN = 200 tokens (percentil 95 del corpus). Evita truncar la mayoría
  de reseñas. Actualizar tras el EDA de Felipe si el valor difiere.

- SPLITS 70/15/15 estratificados: garantiza representación proporcional de
  todas las clases en cada partición. Crítico con dataset desbalanceado.

- SEMILLA 42: Fija en todas las operaciones. NUNCA modificar.
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ──────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE  = 20_000   # Top-N palabras más frecuentes en el corpus
MAX_LEN     = 200      # Longitud fija de secuencia (percentil 95 del EDA)
PAD_TOKEN   = "<PAD>"  # Índice 0  — relleno para padding
UNK_TOKEN   = "<UNK>"  # Índice 1  — palabras fuera del vocabulario
SEED        = 42       # Semilla fija del equipo — NO modificar


# ──────────────────────────────────────────────────────────────────────────────
# 1. Limpieza de texto
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpia un texto individual para su uso en modelos de texto en español.

    Pasos aplicados (en orden):
        1. Convertir a string si no lo es.
        2. Convertir a minúsculas.
        3. Normalizar unicode NFC (preserva tildes y ñ sin alterar).
        4. Eliminar URLs (http://, www.) y direcciones de email.
        5. Eliminar números (no aportan semántica de sentimiento).
        6. Eliminar caracteres especiales; conservar letras, tildes, ñ, ¡, ¿.
        7. Normalizar espacios múltiples a uno solo.
        8. Strip final.

    Decisión sobre tildes: PRESERVADAS. "mas" (conjunción) ≠ "más" (adverbio).
    Decisión sobre stop words: NO eliminadas. "no", "nunca" son críticas en sentimiento.

    Args:
        text (str): Texto crudo de una reseña de hotel.

    Returns:
        str: Texto limpio en minúsculas, sin ruido.

    Ejemplo:
        >>> clean_text("¡Excelente hotel! Ver: http://hotel.com  Tel: 123-456")
        '¡excelente hotel ver'
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-záéíóúüñ¡¿\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tokenizador (vocabulario por frecuencia)
# ──────────────────────────────────────────────────────────────────────────────

def build_tokenizer(corpus: list, vocab_size: int = VOCAB_SIZE) -> dict:
    """
    Construye un diccionario palabra → índice entero a partir del corpus
    de entrenamiento.

    Estrategia: tokenización por palabras, vocabulario limitado a las
    `vocab_size` palabras más frecuentes. Palabras fuera del vocabulario
    se mapean a <UNK> (índice 1) en tiempo de inferencia.

    IMPORTANTE: Llamar SOLO con el corpus de entrenamiento. Nunca incluir
    val o test para evitar data leakage en el vocabulario.

    Reservas especiales:
        - Índice 0: <PAD>  — token de relleno
        - Índice 1: <UNK>  — palabras fuera del vocabulario

    Args:
        corpus (list[str]): Textos ya limpios con clean_text(). Solo train split.
        vocab_size (int)  : Número máximo de palabras (sin <PAD> y <UNK>).

    Returns:
        dict: word2idx — mapeo {palabra: índice_entero}.
    """
    counter = Counter()
    for text in corpus:
        counter.update(text.split())

    most_common = counter.most_common(vocab_size)

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx

    print(f"  Vocabulario construido: {len(word2idx):,} tokens "
          f"(top-{vocab_size} + PAD + UNK)")
    return word2idx


# ──────────────────────────────────────────────────────────────────────────────
# 3. Encoding + padding/truncado
# ──────────────────────────────────────────────────────────────────────────────

def encode_sequences(texts: list, word2idx: dict, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Convierte textos limpios en una matriz de índices con longitud fija.

    Estrategia:
        - Truncado: Se conservan los primeros `max_len` tokens (pre-truncado).
          En reseñas, el juicio principal suele estar al comienzo.
        - Padding: Se rellena con ceros (<PAD>) al final (post-padding).
          Usar padding_idx=0 en nn.Embedding para que el modelo lo ignore.

    Args:
        texts    (list[str]) : Textos limpios.
        word2idx (dict)      : Vocabulario de build_tokenizer().
        max_len  (int)       : Longitud fija de las secuencias de salida.

    Returns:
        np.ndarray: Shape (n_textos, max_len), dtype int32.
    """
    unk_idx = word2idx.get(UNK_TOKEN, 1)
    pad_idx = word2idx.get(PAD_TOKEN, 0)
    sequences = np.full((len(texts), max_len), pad_idx, dtype=np.int32)

    for i, text in enumerate(texts):
        tokens = text.split()[:max_len]
        for j, token in enumerate(tokens):
            sequences[i, j] = word2idx.get(token, unk_idx)

    return sequences


# ──────────────────────────────────────────────────────────────────────────────
# 4. Splits estratificados 70/15/15
# ──────────────────────────────────────────────────────────────────────────────

def build_splits(df: pd.DataFrame, text_col: str, target_col: str,
                 seed: int = SEED, output_dir = "data"):
    """
    Genera splits estratificados 70/15/15 y los guarda como CSV.

    Estratificación: garantiza que la distribución de clases sea proporcional
    en los tres splits. Crítico para el dataset desbalanceado de reseñas.

    Semilla FIJA = 42. NO modificar. Todos los miembros del equipo usan
    estos archivos; nadie genera sus propios splits.

    Args:
        df         (pd.DataFrame): DataFrame completo.
        text_col   (str)         : Columna de texto.
        target_col (str)         : Columna de etiquetas.
        seed       (int)         : Semilla aleatoria (default 42).
        output_dir               : Carpeta destino para los CSV.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df[[text_col, target_col]],
        test_size=0.30,
        stratify=df[target_col],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[target_col],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv",     index=False)
    test_df.to_csv(output_dir / "test.csv",   index=False)

    print(f"  Splits guardados en '{output_dir}/'")
    print(f"    train : {len(train_df):,}   val : {len(val_df):,}   test : {len(test_df):,}")
    print("  Distribución de clases (%):")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = sdf[target_col].value_counts(normalize=True).sort_index()
        line = "  ".join([f"cls {k}: {v:.1%}" for k, v in dist.items()])
        print(f"    {name}: {line}")

    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Class weights para CrossEntropyLoss
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y_train) -> torch.Tensor:
    """
    Calcula pesos de clase inversamente proporcionales a su frecuencia.

    Motivación: El dataset está DESBALANCEADO. Sin class weights, el modelo
    aprende a predecir las clases frecuentes → accuracy alta pero F1 macro
    baja. Los pesos fuerzan al modelo a prestar más atención a las clases
    escasas (2 y 3 estrellas típicamente).

    Fórmula (sklearn 'balanced'):  weight_c = n_total / (n_classes × n_c)

    IMPORTANTE: Calcular SOLO sobre y_train. Nunca incluir val/test.

    Args:
        y_train (array-like): Etiquetas del set de entrenamiento (0-indexed).

    Returns:
        torch.Tensor: Shape (n_classes,). Usar como:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    """
    y_array = np.array(y_train)
    classes = np.unique(y_array)
    weights = compute_class_weight("balanced", classes=classes, y=y_array)

    print("  Class weights (modo 'balanced'):")
    for cls, w in zip(classes, weights):
        print(f"    Clase {cls}: {w:.4f}")

    return torch.tensor(weights, dtype=torch.float)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dataset de PyTorch
# ──────────────────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """
    Dataset de PyTorch para reseñas de hoteles tokenizadas.

    Cada ítem devuelve (secuencia_codificada, etiqueta) como tensores,
    compatible con training.py de Daniel.

    Args:
        sequences (np.ndarray): Array (n, max_len) de índices enteros int32.
        labels    (np.ndarray): Array (n,) de etiquetas enteras (0-indexed).
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Función orquestadora: build_pipeline()
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline(config: dict) -> dict:
    """
    Pipeline completo de preprocesamiento: desde los CSV hasta los DataLoaders.

    Flujo interno:
        1. Carga train.csv, val.csv, test.csv desde data_dir.
        2. Aplica clean_text() a todas las reseñas.
        3. Construye vocabulario (word2idx) SOLO desde train (sin data leakage).
        4. Encodea y hace padding/truncado a las tres particiones.
        5. Convierte etiquetas a 0-indexed (dataset tiene labels 1-5 → 0-4).
        6. Construye ReviewDataset + DataLoader para cada partición.
        7. Calcula class_weights desde y_train.

    Args:
        config (dict): Claves requeridas:
            data_dir   (str)  — carpeta con train/val/test.csv
            text_col   (str)  — columna de texto (ej. "review")
            label_col  (str)  — columna de etiquetas (ej. "rating")
            vocab_size (int)  — tamaño del vocabulario (default 20000)
            max_len    (int)  — longitud de secuencia (default 200)
            batch_size (int)  — tamaño del mini-batch (ej. 64)

    Returns:
        dict con:
            "train_loader"   : DataLoader
            "val_loader"     : DataLoader
            "test_loader"    : DataLoader
            "word2idx"       : dict {palabra: índice}
            "class_weights"  : torch.Tensor (n_classes,)
            "vocab_size"     : int — tamaño real del vocabulario
            "max_len"        : int — longitud de secuencia usada

    Ejemplo:
        >>> pipeline = build_pipeline({
        ...     "data_dir"  : "data",
        ...     "text_col"  : "review",
        ...     "label_col" : "rating",
        ...     "vocab_size": 20000,
        ...     "max_len"   : 200,
        ...     "batch_size": 64,
        ... })
        >>> for X, y in pipeline["train_loader"]:
        ...     print(X.shape, y.shape)
        ...     break
        torch.Size([64, 200]) torch.Size([64])
    """
    data_dir   = Path(config["data_dir"])
    text_col   = config["text_col"]
    label_col  = config["label_col"]
    vocab_size = config.get("vocab_size", VOCAB_SIZE)
    max_len    = config.get("max_len",    MAX_LEN)
    batch_size = config.get("batch_size", 64)

    print("── build_pipeline() ─────────────────────────────────────────")

    # 1. Cargar splits
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")
    print(f"  Splits cargados → train:{len(train_df):,}  val:{len(val_df):,}  test:{len(test_df):,}")

    # 2. Limpiar texto
    print("  Aplicando clean_text()…")
    train_texts = [clean_text(t) for t in train_df[text_col]]
    val_texts   = [clean_text(t) for t in val_df[text_col]]
    test_texts  = [clean_text(t) for t in test_df[text_col]]

    # 3. Construir vocabulario (solo desde train — evita data leakage)
    word2idx = build_tokenizer(train_texts, vocab_size=vocab_size)
    actual_vocab_size = len(word2idx)

    # 4. Encodear secuencias
    print(f"  Encodificando secuencias (max_len={max_len})…")
    X_train = encode_sequences(train_texts, word2idx, max_len)
    X_val   = encode_sequences(val_texts,   word2idx, max_len)
    X_test  = encode_sequences(test_texts,  word2idx, max_len)

    # 5. Etiquetas 0-indexed (dataset 1-5 → 0-4)
    y_train = train_df[label_col].values - 1
    y_val   = val_df[label_col].values   - 1
    y_test  = test_df[label_col].values  - 1

    assert y_train.min() >= 0, "Error: etiquetas negativas. Revisar label_col."

    # 6. Datasets y DataLoaders
    train_loader = DataLoader(ReviewDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ReviewDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(ReviewDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    # 7. Class weights
    class_weights = compute_class_weights(y_train)

    print(f"\n  Pipeline listo:")
    print(f"    Vocab real    : {actual_vocab_size:,} tokens")
    print(f"    max_len       : {max_len}")
    print(f"    batch_size    : {batch_size}")
    print(f"    Batches train : {len(train_loader):,}")
    print("──────────────────────────────────────────────────────────────\n")

    return {
        "train_loader"  : train_loader,
        "val_loader"    : val_loader,
        "test_loader"   : test_loader,
        "word2idx"      : word2idx,
        "class_weights" : class_weights,
        "vocab_size"    : actual_vocab_size,
        "max_len"       : max_len,
    }
