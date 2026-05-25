# Deep Learning RNN — Andalusian Hotels Reviews

**Proyecto Final - Aprendizaje Profundo - Maestria en IA**
**Pontificia Universidad Javeriana - Bogota - 2026**
**Profesor:** Ing. Julio Omar Palacio Nino, M.Sc.
**Entrega y presentacion:** Lunes 25 de mayo de 2026

---

## Equipo

| Persona | Rol | Track |
|---|---|---|
| **Felipe Reyes** | Modelo Clasico 1 (LSTM) + Articulo IEEE | Proyecto Principal |
| **Yibby Gonzalez** | Preprocesamiento + Modelo Clasico 2 (BiLSTM) + Reporte tecnico | Proyecto Principal |
| **Daniel Ruiz** | Esqueleto entrenamiento + Modelo Combinado (BiLSTM+MultiHead) + Presentacion | Proyecto Principal |
| **Sebastian Ruiz** | Modelo Avanzado (BETO fine-tuning) + Informe investigacion | Tarea de Investigacion |

---

## Dataset

**Andalusian Hotels Reviews (Unbalanced)**
URL: https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced

- 18,172 resenas de hoteles en espanol (Andalucia, Espana)
- 5 clases (1-5 estrellas), labels 0-4 en PyTorch
- Dataset **desbalanceado** (ratio 9.1:1 entre clase 5 y clase 2)
- Columnas relevantes: `review_text`, `label`
- Archivo crudo: `data/Big_AHR.csv`
- Splits: train=12,720 / val=2,726 / test=2,726 (70/15/15, estratificados, semilla 42)

**Distribucion de clases (train):**

| Clase | Rating | Muestras | % |
|---|---|---|---|
| 0 | 1 estrella | 1,174 | 9.2% |
| 1 | 2 estrellas | 696 | 5.5% |
| 2 | 3 estrellas | 1,592 | 12.5% |
| 3 | 4 estrellas | 2,955 | 23.2% |
| 4 | 5 estrellas | 6,303 | 49.6% |

**Limitacion conocida — data leakage entre splits:**
- Train/Val comparten 868 resenas, Train/Test comparten 902, Val/Test comparten 249
- Duplicados exactos dentro de train: 2,674 (21%)
- Documentado como limitacion en los entregables escritos

---

## Estructura del repositorio

```
Proyecto_Deep_Learning_Final/
  data/
    Big_AHR.csv                          # Dataset completo
    train.csv / val.csv / test.csv       # Splits estratificados (semilla 42)
  src/
    preprocessing.py                     # Pipeline de texto (Yibby)
    training.py                          # Esqueleto de entrenamiento (Daniel)
    metrics.py                           # Metricas y visualizacion (Daniel)
  notebooks/
    01_eda_felipe.ipynb                  # EDA dimensionalidad y clases [PP]
    02_eda_yibby.ipynb                   # EDA features y texto [PP]
    03_lstm_felipe.ipynb                 # Modelo Clasico 1: LSTM [PP]
    04_bilstm_yibby.ipynb                # Modelo Clasico 2: BiLSTM [PP]
    05_combined_daniel.ipynb             # Modelo Combinado: BiLSTM+MultiHead [PP]
    06_transformer_sebas.ipynb           # Modelo Avanzado: BETO fine-tuning [TI]
  results/                              # JSONs de metricas + checkpoints
  figures/                              # Graficas exportadas (PNG)
  docs/                                 # Borradores de entregables escritos
  base_proyecto/                        # PDFs originales del profesor
```

---

## Modelos y resultados

### PROYECTO PRINCIPAL (PP) — 3 modelos

| Modelo | Autor | F1 macro | Accuracy | Params | Tiempo |
|---|---|---|---|---|---|
| LSTM v3 (Clasico 1) | Felipe | 0.3996 | 0.6042 | 2,700,933 | 321s CPU |
| BiLSTM v1 (Clasico 2) | Yibby | 0.5749 | 0.6871 | 3,253,253 | 27,686s CPU |
| **BiLSTM+MultiHead (Combinado)** | Daniel | **0.6193** | 0.6548 | 6,110,981 | 304s GPU |

### TAREA DE INVESTIGACION (TI) — 1 modelo

| Modelo | Autor | F1 macro | Accuracy | Params entrenables | Tiempo |
|---|---|---|---|---|---|
| **BETO v3 fine-tuned** | Sebastian | **0.6565** | **0.7252** | 21,858,053 (19.9%) | 1,780s GPU |

### F1 por clase (modelos definitivos)

| Clase | LSTM v3 | BiLSTM v1 | BiLSTM+MultiHead | BETO v3 |
|---|---|---|---|---|
| 1 estrella | 0.686 | 0.716 | 0.756 | **0.777** |
| 2 estrellas | 0.168 | 0.202 | **0.533** | 0.502 |
| 3 estrellas | 0.359 | 0.564 | 0.549 | **0.595** |
| 4 estrellas | 0.022 | **0.572** | 0.480 | 0.561 |
| 5 estrellas | 0.764 | 0.821 | 0.778 | **0.848** |

**Mejor modelo global:** BETO v3 (F1=0.6565)
**Mejor modelo clasico:** BiLSTM v1 (F1=0.5749)

---

## Decisiones tecnicas

| Parametro | Valor | Justificacion |
|---|---|---|
| Framework | PyTorch | Compatibilidad con HuggingFace para transformer |
| Semilla | 42 | Reproducibilidad entre miembros |
| VOCAB_SIZE | 20,000 | 99.2% cobertura del corpus limpio |
| MAX_LEN (clasicos) | 150-200 | P95 del EDA = 148 tokens |
| MAX_LEN (transformer) | 128 | Subword tokens BETO, ~100-150 palabras |
| Stop words | CONSERVADAS | "no", "muy", "poco" criticas para sentimiento |
| Tildes | PRESERVADAS | "mas" vs "mas" diferencia semantica en espanol |
| Embeddings clasicos | Entrenables desde cero | Sin redes pre-entrenadas (restriccion del profesor) |
| Transformer base | BETO cased | dccuchile/bert-base-spanish-wwm-cased |
| Class weights | Varian por modelo | Ver documentos fuente en docs/ |

---

## Reproducibilidad

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import pandas as pd
train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")
test_df  = pd.read_csv("data/test.csv")
```

**Los splits se cargan, no se regeneran.**

---

## Como entrenar un modelo clasico

```python
from src.training import set_seed, train, evaluate
from src.metrics  import finalize_and_save
from src.preprocessing import build_pipeline

set_seed(42)

pipeline = build_pipeline({
    "data_dir": "data", "text_col": "review_text",
    "label_col": "label", "vocab_size": 20000,
    "max_len": 200, "batch_size": 64
})

config = {
    "model_name": "mi_modelo", "owner": "Nombre", "track": "PP",
    "n_epochs": 20, "lr": 1e-3, "patience": 5,
    "checkpoint_path": "results/mi_modelo_best.pt",
    "use_class_weights": False,
    "embedding_dim": 128, "hidden_size": 128, "dropout": 0.3
}

metrics_dict = train(model, pipeline["train_loader"], pipeline["val_loader"], config)
y_true, y_pred = evaluate(model, pipeline["test_loader"], config["checkpoint_path"], config)
metrics_dict = finalize_and_save(y_true, y_pred, metrics_dict, show_plots=True)
```

---

## Hitos

| Hito | Fecha | Estado |
|---|---|---|
| Hito 1: Cimientos listos | Vie 8 mayo | COMPLETADO |
| Hito 2: Modelos congelados | Jue 14 mayo | COMPLETADO |
| Hito 3: Borradores 1.0 | Jue 21 mayo | ATRASADO (2/4 borradores) |
| **Entrega y presentacion** | **Lun 25 mayo** | **HOY** |
