# Deep Learning RNN — Andalusian Hotels Reviews

**Proyecto Final - Aprendizaje Profundo - Maestria en IA**
**Pontificia Universidad Javeriana - Bogota - 2026**
**Profesor:** Ing. Julio Omar Palacio Nino, M.Sc.
**Entrega y presentacion:** Lunes 25 de mayo de 2026

---

## Equipo

| Persona | Rol | Track |
|---|---|---|
| **Felipe Reyes** | Modelo Clasico 1 (LSTM) - Articulo IEEE | Proyecto Principal |
| **Yibby Gonzalez** | Preprocesamiento - Modelo Clasico 2 (BiLSTM) - Reporte tecnico | Proyecto Principal |
| **Daniel Ruiz** | Esqueleto entrenamiento - Modelo Nuevo/Combinado - Presentacion | Proyecto Principal |
| **Sebastian Ruiz** | Modelo Avanzado (Transformer fine-tuning) - Informe investigacion | Tarea de Investigacion |

---

## Dataset

**Andalusian Hotels Reviews (Unbalanced)**
URL: https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced

- 18,172 resenas de hoteles en espanol
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
- Documentar como limitacion en articulo IEEE y reporte tecnico

---

## Inventario de archivos (actualizado 24 mayo 2026)

### Data

| Archivo | Estado | Notas |
|---|---|---|
| `data/Big_AHR.csv` | LISTO | Dataset completo (18,172 filas, 7 columnas) |
| `data/train.csv` | LISTO | 12,720 muestras, columnas: review_text, label (0-4) |
| `data/val.csv` | LISTO | 2,726 muestras |
| `data/test.csv` | LISTO | 2,726 muestras |

### Codigo fuente (src/)

| Archivo | Estado | Autor | Funciones |
|---|---|---|---|
| `src/preprocessing.py` | LISTO | Yibby | clean_text, build_tokenizer, encode_sequences, build_splits, compute_class_weights, ReviewDataset, build_pipeline |
| `src/training.py` | LISTO | Daniel | set_seed, EarlyStopping, train, evaluate |
| `src/metrics.py` | LISTO | Daniel | compute_metrics, save_metrics_json, plot_training_curves, plot_confusion_matrix, finalize_and_save |
| `src/__init__.py` | LISTO | - | Archivo vacio para importacion |

### Notebooks

| Archivo | Estado | Autor | Ejecutado con outputs |
|---|---|---|---|
| `notebooks/01_eda_felipe.ipynb` | LISTO | Felipe | Si - 12/12 items EDA completos |
| `notebooks/02_eda_yibby.ipynb` | LISTO | Yibby/Felipe | Si - 7/7 items EDA completos |
| `notebooks/03_lstm_felipe.ipynb` | LISTO | Felipe | Si - 4 versiones entrenadas, v3 definitiva |
| `notebooks/04_bilstm_yibby.ipynb` | LISTO | Yibby/Felipe | Si - 2 versiones entrenadas, v1 definitiva |
| `notebooks/05_combined_daniel.ipynb` | **PENDIENTE** | Daniel | Solo tiene celda de configuracion y un TODO |
| `notebooks/06_transformer_sebas.ipynb` | **PARCIAL** | Sebastian | Solo 1 epoca local (fallo espacio disco MPS) |
| `PreEntrenado.ipynb` | LISTO | Sebastian | Ejecutado en Colab con GPU (5 epocas, Tesla T4) |

### Resultados (results/)

| Archivo | Estado | Modelo | F1 macro | Accuracy |
|---|---|---|---|---|
| `results/lstm_metrics.json` | LISTO | LSTM v3 (definitivo) | **0.3996** | 0.6042 |
| `results/lstm_v2_metrics.json` | LISTO | LSTM v2 (historico) | 0.3830 | 0.5675 |
| `results/lstm_v3_yibby_metrics.json` | LISTO | LSTM v3 (copia) | 0.3996 | 0.6042 |
| `results/lstm_v4_sqrt_metrics.json` | LISTO | LSTM v4 sqrt (historico) | 0.3919 | 0.5829 |
| `results/bilstm_metrics.json` | LISTO | BiLSTM v1 (definitivo) | **0.5568** | 0.6915 |
| `results/bilstm_v1_metrics.json` | LISTO | BiLSTM v1 (copia) | 0.5568 | 0.6915 |
| `results/bilstm_v2_metrics.json` | LISTO | BiLSTM v2 (historico) | 0.5224 | 0.5594 |
| `results/combined_metrics.json` | **FALTA** | Modelo Nuevo/Combinado | - | - |
| `results/transformer_metrics.json` | **FALTA** | BETO fine-tuned | - | - |

**Nota sobre transformer:** Sebastian ejecuto BETO en Colab (notebook `PreEntrenado.ipynb`) obteniendo **F1 macro=0.5873, Accuracy=0.6592** (5 epocas, 684s en Tesla T4). Los resultados se guardaron en Google Drive (`beto_finetuned_metrics.json`) pero **no estan en el repositorio local**.

### Figuras (figures/)

| Archivo | Estado |
|---|---|
| `figures/eda_distribucion_clases.png` | LISTO |
| `figures/eda_histograma_longitudes.png` | LISTO |
| `figures/eda_boxplot_longitud_clase.png` | LISTO |
| `figures/eda_zipf_cobertura.png` | LISTO |
| `figures/eda_top50_words_global.png` | LISTO |
| `figures/eda_top30_por_clase.png` | LISTO |
| `figures/eda_wordclouds_por_clase.png` | LISTO |
| `figures/eda_class_distribution_weights.png` | LISTO |
| `figures/lstm_curves.png` | LISTO |
| `figures/lstm_confusion_matrix.png` | LISTO |
| `figures/lstm_v2_curves.png` | LISTO |
| `figures/lstm_v2_confusion_matrix.png` | LISTO |
| `figures/lstm_v3_yibby_curves.png` | LISTO |
| `figures/lstm_v3_yibby_confusion_matrix.png` | LISTO |
| `figures/lstm_v4_sqrt_curves.png` | LISTO |
| `figures/lstm_v4_sqrt_confusion_matrix.png` | LISTO |
| `figures/bilstm_v1_curves.png` | LISTO |
| `figures/bilstm_v1_confusion_matrix.png` | LISTO |
| `figures/bilstm_v2_curves.png` | LISTO |
| `figures/bilstm_v2_confusion_matrix.png` | LISTO |
| `figures/bilstm_analisis_final.png` | LISTO |
| `figures/combined_curves.png` | **FALTA** |
| `figures/combined_confusion_matrix.png` | **FALTA** |
| `figures/transformer_curves.png` | **FALTA** (existe en Drive de Sebastian) |
| `figures/transformer_confusion_matrix.png` | **FALTA** (existe en Drive de Sebastian) |

### Documentos (docs/)

| Documento | Estado | Responsable |
|---|---|---|
| Articulo IEEE (4-6 pag, formato IEEE Conference) | **FALTA** | Felipe |
| Reporte tecnico detallado (sin limite pag) | **FALTA** | Yibby |
| Presentacion de slides | **FALTA** | Daniel |
| Informe de investigacion (max 5 pag) | **FALTA** | Sebastian |

---

## Tabla comparativa de modelos (estado actual)

| Modelo | Owner | Track | F1 macro | Accuracy | Params | Tiempo (s) | Estado |
|---|---|---|---|---|---|---|---|
| BiLSTM v1 | Yibby | PP | **0.5568** | 0.6915 | 3,253,253 | 381.9 | Definitivo |
| BETO fine-tuned | Sebastian | TI | **0.5873** | 0.6592 | 109,854,725 (7.6M entrenables) | 684 | Ejecutado en Colab, falta integrar JSON al repo |
| LSTM v3 | Felipe | PP | **0.3996** | 0.6042 | 2,700,933 | 321.2 | Definitivo |
| Modelo Nuevo/Combinado | Daniel | PP | - | - | - | - | **NO INICIADO** |

**Mejor modelo clasico:** BiLSTM v1 de Yibby (F1 macro = 0.5568)

**Decisiones tecnicas confirmadas:**
- Framework: PyTorch (todo el equipo)
- Transformer: BETO cased (`dccuchile/bert-base-spanish-wwm-cased`)
- Capas descongeladas: encoder.layer.11 + pooler + classifier (7.0% de params)
- Embeddings clasicos: entrenables desde cero (no pre-cargados)
- VOCAB_SIZE: 20,000 (99.2% cobertura)
- MAX_LEN: 200 (preprocessing.py default), 150 (LSTM de Felipe)
- Stop words: CONSERVADAS (criticas para sentimiento en espanol)
- Tildes: PRESERVADAS

---

## Reproducibilidad

**Semilla fija = 42 en todo.** La primera celda de cada notebook debe ser:

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

**Los splits se cargan desde `data/` (no se regeneran):**

```python
import pandas as pd
train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")
test_df  = pd.read_csv("data/test.csv")
```

---

## Como entrenar un modelo

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

**Nota:** Para usar class weights, agregar `"class_weights": pipeline["class_weights"]` al config.

---

## Formato JSON de metricas (estandarizado)

```json
{
  "model_name": "lstm",
  "owner": "Felipe",
  "track": "PP",
  "config": { "embedding_dim": 128, "hidden_size": 128, "dropout": 0.3,
              "use_class_weights": false, "n_params": 2700933 },
  "metrics": {
    "accuracy": 0.6042, "precision_macro": 0.5639, "recall_macro": 0.4408, "f1_macro": 0.3996,
    "f1_per_class": {"1": 0.686, "2": 0.168, "3": 0.359, "4": 0.022, "5": 0.764},
    "confusion_matrix": [[...]]
  },
  "training": {
    "epochs_run": 24, "best_epoch": 19, "training_time_seconds": 321.2,
    "loss_history": [...], "val_loss_history": [...],
    "acc_history": [...], "val_acc_history": [...]
  }
}
```

---

## Hitos

| Hito | Fecha | Estado |
|---|---|---|
| Hito 1: Cimientos listos | Vie 8 mayo | COMPLETADO |
| Hito 2: Modelos congelados + metricas | Jue 14 mayo | PARCIAL (falta combined de Daniel) |
| Hito 3: Borradores 1.0 completos | Jue 21 mayo | NO COMPLETADO (0/4 documentos) |
| Entrega y presentacion | **Lun 25 mayo** | PENDIENTE |

---

## Convenciones

- **Metricas:** siempre F1 macro y por clase. Nunca solo accuracy (dataset desbalanceado).
- **Test set:** solo se mira con `evaluate()`, al final, **una vez por modelo**.
- **Checkpoints:** `torch.save(model.state_dict(), path)` -- siempre.
- **Antes de push:** ejecutar el notebook completo y dejar las salidas guardadas.
- **Columnas CSV:** `review_text` (texto) y `label` (0-4, ya 0-indexed).
