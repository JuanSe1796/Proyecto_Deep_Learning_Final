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

## Inventario de archivos (revalidado 25 mayo 2026)

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
| `notebooks/05_combined_daniel.ipynb` | LISTO | Daniel | Si - 3 variantes entrenadas en GPU (CUDA), BiLSTM+MultiHead definitivo |
| `notebooks/06_transformer_sebas.ipynb` | LISTO | Sebastian | Ejecutado en GPU local (CUDA, PyTorch 2.12). Notebook CANONICO del transformer — 3 variantes entrenadas (v1/v2/v3), v3 definitiva (F1=0.6565). Incluye pre-tokenizacion, early stopping, LR discriminativo, comparacion final automatica |

**Nota:** `PreEntrenado.ipynb` era un borrador anterior de Sebastian ejecutado en Colab. No es necesario para reproducibilidad — todo su contenido relevante esta consolidado en `06_transformer_sebas.ipynb`.

### Resultados (results/)

**Modelos definitivos:**

| Archivo | Estado | Modelo | F1 macro (JSON) | Accuracy (JSON) |
|---|---|---|---|---|
| `results/bilstm_multihead_metrics.json` | LISTO | BiLSTM+MultiHead (Combinado definitivo) | **0.6193** | 0.6548 |
| `results/transformer_metrics.json` | LISTO | BETO fine-tuned v3 (definitivo) | **0.6565** | 0.7252 |
| `results/bilstm_v1_metrics.json` | LISTO | BiLSTM v1 (Clasico 2 definitivo, sin class weights) | **0.5749** | 0.6871 |
| `results/lstm_metrics.json` | LISTO | LSTM v3 (Clasico 1 definitivo) | **0.3996** | 0.6042 |

**Variantes exploratorias (historico):**

| Archivo | Estado | Modelo | F1 macro | Accuracy |
|---|---|---|---|---|
| `results/bilstm_metrics.json` | LISTO | BiLSTM v2 (con class weights) — NOTA: archivo etiquetado como "bilstm" pero contiene v2 | 0.5989 | 0.6453 |
| `results/gru_bahdanau_metrics.json` | LISTO | GRU+Bahdanau (Daniel, variante 2) | 0.5785 | 0.6416 |
| `results/bilstm_bahdanau_metrics.json` | LISTO | BiLSTM+Bahdanau (Daniel, variante 1) | 0.5740 | 0.6471 |
| `results/bilstm_v2_metrics.json` | LISTO | BiLSTM v2 con class weights (identico a bilstm_metrics.json) | 0.5989 | 0.6453 |
| `results/lstm_v2_metrics.json` | LISTO | LSTM v2 (historico) | 0.3830 | 0.5675 |
| `results/lstm_v3_yibby_metrics.json` | LISTO | LSTM v3 (copia) | 0.3996 | 0.6042 |
| `results/lstm_v4_sqrt_metrics.json` | LISTO | LSTM v4 sqrt (historico) | 0.3919 | 0.5829 |
| `results/tabla_comparativa.csv` | LISTO (con errores) | Tabla comparativa generada por notebook 05 — ver seccion de bugs | - | - |

**Checkpoints (.pt):**

| Archivo | Estado | Notas |
|---|---|---|
| `results/lstm_best.pt` | LISTO | LSTM v3 definitivo (10.3MB) |
| `results/lstm_v2_best.pt` | LISTO | Historico |
| `results/lstm_v3_best.pt` | LISTO | Historico |
| `results/lstm_v4_best.pt` | LISTO | Historico |
| `results/bilstm_best.pt` | LISTO | BiLSTM (13MB) |
| `results/bilstm_v1_best.pt` | LISTO | BiLSTM v1 (13MB) |
| `results/bilstm_v2_best.pt` | LISTO | Historico |
| `results/bilstm_bahdanau_best.pt` | LISTO | 20.7MB — movido de raiz a results/ |
| `results/gru_bahdanau_best.pt` | **FALTA** | Generado en GPU, no copiado al repo |
| `results/bilstm_multihead_best.pt` | **FALTA** | Checkpoint del modelo DEFINITIVO, generado en GPU, no copiado |
| `results/transformer_best.pt` | **FALTA** | Checkpoint v3 definitivo, generado en GPU local (~440MB). Copiar al repo |
| `results/transformer_v2_best.pt` | **FALTA** | Checkpoint v2, generado en GPU local |
| `results/transformer_v3_best.pt` | **FALTA** | Checkpoint v3 (copia), generado en GPU local |

### Figuras (figures/)

**EDA (8 figuras):**

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

**Modelos clasicos — LSTM (8 figuras, 4 versiones x 2):**

| Archivo | Estado |
|---|---|
| `figures/lstm_curves.png` | LISTO |
| `figures/lstm_confusion_matrix.png` | LISTO |
| `figures/lstm_v2_curves.png` | LISTO |
| `figures/lstm_v2_confusion_matrix.png` | LISTO |
| `figures/lstm_v3_yibby_curves.png` | LISTO |
| `figures/lstm_v3_yibby_confusion_matrix.png` | LISTO |
| `figures/lstm_v4_sqrt_curves.png` | LISTO |
| `figures/lstm_v4_sqrt_confusion_matrix.png` | LISTO |

**Modelos clasicos — BiLSTM (5 figuras):**

| Archivo | Estado |
|---|---|
| `figures/bilstm_v1_curves.png` | LISTO |
| `figures/bilstm_v1_confusion_matrix.png` | LISTO |
| `figures/bilstm_v2_curves.png` | LISTO |
| `figures/bilstm_v2_confusion_matrix.png` | LISTO |
| `figures/bilstm_analisis_final.png` | LISTO |

**Modelo combinado — Daniel (7 figuras):**

| Archivo | Estado |
|---|---|
| `figures/bilstm_bahdanau_curves.png` | LISTO |
| `figures/bilstm_bahdanau_confusion_matrix.png` | LISTO |
| `figures/gru_bahdanau_curves.png` | LISTO |
| `figures/gru_bahdanau_confusion_matrix.png` | LISTO |
| `figures/bilstm_multihead_curves.png` | LISTO |
| `figures/bilstm_multihead_confusion_matrix.png` | LISTO |
| `figures/combined_f1_por_clase.png` | LISTO |

**Transformer — Sebastian (3 figuras):**

| Archivo | Estado | Nota |
|---|---|---|
| `figures/transformer_curves.png` | LISTO | Curvas de v1 (baseline) |
| `figures/transformer_confusion.png` | LISTO | Confusion normalizada de v1. Nombre difiere de convencion (sin _matrix) |
| `figures/transformer_confusion_absolutos.png` | LISTO | Confusion con valores absolutos de v1 |

### Documentos (docs/)

| Documento | Estado | Responsable |
|---|---|---|
| `docs/articulo_ieee_borrador.md` | BORRADOR ~85% (falta seccion IV-D combinado, Abstract, formateo PDF) | Felipe |
| Articulo IEEE final (PDF, formato IEEE Conference) | **FALTA** | Felipe |
| Reporte tecnico detallado (PDF, sin limite pag) | **FALTA — NO INICIADO** | Yibby |
| Presentacion de slides (.pptx + PDF) | **FALTA — NO INICIADO** | Daniel |
| `docs/Informe_Investigacion.docx` | BORRADOR (en Word, falta PDF final) | Sebastian |

### Archivos de planificacion y temporales

| Archivo | Estado | Notas |
|---|---|---|
| `plan_proyecto_deep_learning.md` | LISTO | Plan maestro del proyecto (fuente de verdad) |
| `plan_avance_global.md` | LISTO | Plan de avance consolidado con estado por persona |
| `freyesTemp/plan_avance.md` | LISTO | Bitacora detallada de Felipe (4 fases documentadas) |
| `freyesTemp/setup_transformer_gpu.md` | LISTO | Guia Miniconda + GPU para notebook transformer |

### Otros archivos

| Archivo | Estado | Notas |
|---|---|---|
| `requirements.txt` | LISTO | Dependencias del proyecto (version pinning flexible con >=) |
| `test.py` | LISTO | Script de inspeccion rapida de datos (no es un test suite real) |
| `setup_transformer_gpu.md` | LISTO | Copia de la guia GPU en raiz del proyecto |
| `.gitignore` | LISTO | Configurado para __pycache__, checkpoints grandes, .env |

---

## BUGS CORREGIDOS (24 mayo 2026)

### BUG 1 (CORREGIDO): Metricas BiLSTM inconsistentes entre documentos y JSONs

Los documentos reportaban F1=0.5568 para BiLSTM v1, pero `bilstm_v1_metrics.json` dice F1=0.5749.
**Correccion:** Todos los documentos (README, plan_avance_global, tabla_comparativa.csv) ahora usan los valores del JSON como fuente de verdad: F1=0.5749, Accuracy=0.6871.

### BUG 2 (CORREGIDO): bilstm_metrics.json contenia datos de v2, no v1

`bilstm_metrics.json` contenia una copia de v2 (con class weights, F1=0.5989).
**Correccion:** Sobreescrito con los datos de bilstm_v1_metrics.json (modelo definitivo, sin class weights, F1=0.5749).

### BUG 3 (CORREGIDO): tabla_comparativa.csv tenia valores 0.0 para LSTM

La fila del LSTM mostraba F1 cl.2=0.0 y F1 cl.3=0.0.
**Correccion:** Valores actualizados a F1 cl.2=0.1675 y F1 cl.3=0.3586 segun lstm_metrics.json. Fila de BiLSTM tambien corregida.

### BUG 4 (CORREGIDO): Checkpoint bilstm_bahdanau_best.pt en ubicacion incorrecta

Existia en la raiz del proyecto (20.7MB).
**Correccion:** Movido a `results/bilstm_bahdanau_best.pt`.

### BUG 5 (CORREGIDO): Keys de F1 por clase inconsistentes entre JSONs

`lstm_metrics.json` usaba keys con estrella ("1★") mientras los demas usaban sin estrella ("1").
**Correccion:** Keys de lstm_metrics.json normalizadas a formato sin estrella ("1", "2", "3", "4", "5").

### BUG 6 (CORREGIDO): Transformer posiblemente no convergio

**Problema original:** best_epoch = 5 = epochs_run = 5. Val loss seguia bajando. No habia early stopping.
**Correccion:** Re-entrenado con 15 epocas max + early stopping (paciencia=3). Se entrenaron 3 variantes:
- v1 (1 capa, balanced): 7 epocas, best=4, F1=0.5926
- v2 (3 capas, balanced, lr discrim.): 9 epocas, best=6, F1=0.6161
- v3 (3 capas, sqrt, lr discrim.): 11 epocas, best=8, F1=0.6565 ← DEFINITIVO
El early stopping confirma convergencia real en las 3 variantes.

### BUG 7 (CORREGIDO): Notebook 05 tenia celda "TODO" residual y visualizacion de atencion rota

**Correccion aplicada en notebook 05_combined_daniel.ipynb:**
- Celda markdown "TODO: Daniel Ruiz completa este notebook" reemplazada por descripcion real del contenido.
- Celda de visualizacion de atencion corregida: eliminado bloque muerto con hidden_size=128 incorrecto, best_model_id fijado a 'bilstm_multihead' (ganador, F1=0.6193) en vez de sobreescribirse con 'bilstm_bahdanau', hiperparametros alineados con el modelo entrenado (hidden_size=256, dropout=0.4, num_heads=4).

### BUG 8 (CORREGIDO — documentado como limitacion): MAX_LEN varia entre modelos

LSTM usa MAX_LEN=150 (Felipe paso `max_len=150` a `build_pipeline()`), BiLSTM/Combinado usan 200 (default), Transformer usa 128 (subword tokens de BETO, no comparable directamente). No se puede corregir sin reentrenar los modelos.
**Correccion:** Tabla detallada de MAX_LEN por modelo agregada a la seccion de Reproducibilidad, con justificacion por modelo y analisis de impacto.

### BUG 9 (CORREGIDO): PreEntrenado.ipynb referenciado como faltante pero no es necesario

`PreEntrenado.ipynb` era un borrador anterior de Sebastian ejecutado en Colab. El notebook canonico `06_transformer_sebas.ipynb` ya contiene la implementacion completa del transformer (setup, clonado del repo, carga de splits, entrenamiento, evaluacion, metricas JSON, figuras).
**Correccion:** Eliminada la entrada de PreEntrenado.ipynb del inventario de notebooks. Agregada nota aclarando que 06 es el notebook canonico y PreEntrenado.ipynb no es necesario para reproducibilidad.

---

## Tabla comparativa de modelos (CORREGIDA - valores de JSONs)

| # | Modelo | Owner | Track | F1 macro | Accuracy | Prec macro | Recall macro | Params | Tiempo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **BETO v3 fine-tuned** | Sebastian | TI | **0.6565** | **0.7252** | 0.6544 | 0.6609 | 21.9M entrenables | 1,780s GPU |
| 2 | **BiLSTM+MultiHead** | Daniel | PP | **0.6193** | 0.6548 | 0.6086 | 0.6454 | 6,110,981 | 303.8s GPU |
| 3 | BETO v2 fine-tuned | Sebastian | TI | 0.6161 | 0.6864 | 0.6084 | 0.6306 | 21.9M entrenables | 1,566s GPU |
| 4 | BiLSTM v2 (class weights) | Yibby | PP | 0.5989 | 0.6453 | 0.5870 | 0.6251 | 3,253,253 | 1,792s CPU |
| 5 | BETO v1 fine-tuned | Sebastian | TI | 0.5926 | 0.6706 | 0.5837 | 0.6106 | 7.68M entrenables | 1,036s GPU |
| 6 | GRU+Bahdanau | Daniel | PP | 0.5785 | 0.6416 | 0.5757 | 0.6024 | 4,599,813 | 119.4s GPU |
| 7 | BiLSTM+Bahdanau | Daniel | PP | 0.5740 | 0.6471 | 0.5836 | 0.5842 | 5,191,685 | 129.3s GPU |
| 8 | BiLSTM v1 (sin weights) | Yibby | PP | 0.5749 | 0.6871 | 0.5876 | 0.5795 | 3,253,253 | 27,686s CPU |
| 9 | LSTM v3 | Felipe | PP | 0.3996 | 0.6042 | 0.5639 | 0.4408 | 2,700,933 | 321.2s CPU |

**NOTA:** BETO v3 supera a todos los modelos del proyecto (F1=0.6565 vs BiLSTM+MultiHead F1=0.6193, +3.7 puntos). Es el mejor en F1 macro, accuracy, y en 4 de 5 clases individuales (1★, 3★, 4★, 5★). BiLSTM+MultiHead mantiene ventaja leve en clase 2★ (0.533 vs 0.502).

**F1 por clase (modelos definitivos, segun JSONs):**

| Clase | LSTM v3 | BiLSTM v1 | BiLSTM+MultiHead | BETO v3 |
|---|---|---|---|---|
| 1 estrella | 0.686 | 0.716 | 0.756 | **0.777** |
| 2 estrellas | 0.168 | 0.202 | 0.533 | **0.502** |
| 3 estrellas | 0.359 | 0.564 | 0.549 | **0.595** |
| 4 estrellas | 0.022 | 0.572 | 0.480 | **0.561** |
| 5 estrellas | 0.764 | 0.821 | 0.778 | **0.848** |

**Mejor global:** BETO v3 de Sebastian (F1 macro = 0.6565, MEJOR de todo el proyecto)
**Modelo combinado seleccionado:** BiLSTM+MultiHead de Daniel (F1 macro = 0.6193, segundo mejor)
**Mejor modelo clasico (segun JSON):** BiLSTM v1 de Yibby (F1 macro = 0.5749) — o v2 (F1=0.5989) si se prefiere F1 sobre accuracy

**Decisiones tecnicas confirmadas:**
- Framework: PyTorch (todo el equipo)
- Modelo combinado: BiLSTM + Multi-Head Attention (4 cabezas), con class weights
- Transformer: BETO cased (`dccuchile/bert-base-spanish-wwm-cased`), v3 definitiva
- Capas descongeladas (v3): encoder.layer.9, 10, 11 + pooler + classifier (19.9% de params)
- Class weights transformer (v3): sqrt(balanced)
- LR discriminativo (v3): layer.9=2e-6, layer.10=1e-5, layer.11+classifier=2e-5
- Embeddings clasicos: entrenables desde cero (no pre-cargados)
- VOCAB_SIZE: 20,000 (99.2% cobertura)
- MAX_LEN: 200 (preprocessing.py default), 150 (LSTM de Felipe), 128 (Transformer)
- Stop words: CONSERVADAS (criticas para sentimiento en espanol)
- Tildes: PRESERVADAS

**Comparabilidad confirmada:** Sebastian uso los mismos splits del repo (data/train.csv, val.csv, test.csv)
clonando el repositorio en Colab. Las metricas del transformer son directamente comparables
con las de los modelos clasicos (mismos 12,720/2,726/2,726 ejemplos con identica distribucion).

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

**Limitacion conocida — MAX_LEN diferente entre modelos:**

| Modelo | MAX_LEN | Tipo de token | Justificacion |
|---|---|---|---|
| LSTM (Felipe) | 150 | palabra | P95 del EDA = 148, redondeado. Felipe paso `max_len=150` a `build_pipeline()` |
| BiLSTM (Yibby) | 200 | palabra | Default del pipeline de Yibby |
| Combinado (Daniel) | 200 | palabra | Mismo pipeline que Yibby |
| Transformer (Sebastian) | 128 | subword (BPE) | Tokenizer propio de BETO. 128 subword tokens cubren ~100-150 palabras |

Esto significa que el LSTM proceso secuencias truncadas a 150 tokens mientras BiLSTM y Combinado procesaron hasta 200. La diferencia es moderada (solo afecta resenas >150 palabras, ~5% del dataset). El Transformer usa tokenizacion diferente (subwords) por lo que su MAX_LEN no es directamente comparable con los modelos clasicos.

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
| Hito 2: Modelos congelados + metricas | Jue 14 mayo | COMPLETADO (4/4 modelos con metricas) |
| Hito 3: Borradores 1.0 completos | Jue 21 mayo | ATRASADO (2/4 borradores: articulo IEEE ~85%, informe investigacion en Word) |
| Entrega y presentacion | **Lun 25 mayo** | CRITICO — faltan documentos finales en PDF |

---

## Convenciones

- **Metricas:** siempre F1 macro y por clase. Nunca solo accuracy (dataset desbalanceado).
- **Test set:** solo se mira con `evaluate()`, al final, **una vez por modelo**.
- **Checkpoints:** `torch.save(model.state_dict(), path)` -- siempre.
- **Antes de push:** ejecutar el notebook completo y dejar las salidas guardadas.
- **Columnas CSV:** `review_text` (texto) y `label` (0-4, ya 0-indexed).
