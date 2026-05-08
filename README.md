# Deep Learning RNN — Andalusian Hotels Reviews

**Proyecto Final · Aprendizaje Profundo · Maestría en IA**  
**Pontificia Universidad Javeriana · Bogotá · 2026**  
**Profesor:** Ing. Julio Omar Palacio Niño, M.Sc.

---

## Equipo

| Persona | Rol | Track |
|---|---|---|
| **Felipe Reyes** | Modelo Clásico 1 (LSTM/GRU) · Artículo IEEE | Proyecto Principal |
| **Yibby González** | Preprocesamiento · Modelo Clásico 2 (BiLSTM) · Reporte técnico | Proyecto Principal |
| **Daniel Ruiz** | Esqueleto entrenamiento · Modelo Nuevo/Combinado · Presentación | Proyecto Principal |
| **Sebastián Ruiz** | Modelo Avanzado (Transformer fine-tuning) · Informe investigación | Tarea de Investigación |

---

## Dataset

**Andalusian Hotels Reviews (Unbalanced)**  
URL: https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced  
- Reseñas de hoteles en español · 5 clases (1–5 estrellas) · Dataset desbalanceado

---

## Estructura del repositorio

```
deep-learning-rnn/
  data/
    train.csv               ← Yibby genera (semilla 42, estratificado 70%)
    val.csv                 ← Yibby genera (semilla 42, estratificado 15%)
    test.csv                ← Yibby genera (semilla 42, estratificado 15%)
  notebooks/
    01_eda_felipe.ipynb     ← EDA dimensionalidad + variable objetivo
    02_eda_yibby.ipynb      ← EDA features, texto, preprocesamiento
    03_lstm_felipe.ipynb    ← Modelo Clásico 1 (LSTM/GRU)
    04_bilstm_yibby.ipynb   ← Modelo Clásico 2 (BiLSTM)
    05_combined_daniel.ipynb← Modelo Nuevo/Combinado (BiLSTM + Atención)
    06_transformer_sebas.ipynb ← Modelo Avanzado — Fine-tuning (Tarea Investigación)
  src/
    preprocessing.py        ← Pipeline de texto reusable (Yibby)
    training.py             ← Esqueleto train() reusable (Daniel)
    metrics.py              ← JSON estandarizado + gráficas (Daniel)
  results/
    lstm_metrics.json       ← Métricas finales Clásico 1
    bilstm_metrics.json     ← Métricas finales Clásico 2
    combined_metrics.json   ← Métricas finales Nuevo/Combinado
    transformer_metrics.json← Métricas finales Avanzado
  figures/
    *_curves.png            ← Curvas loss/accuracy por modelo
    *_confusion_matrix.png  ← Matrices de confusión por modelo
  docs/                     ← Borradores de entregables escritos
  README.md
  requirements.txt
  .gitignore
```

---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/deep-learning-rnn.git
cd deep-learning-rnn
pip install -r requirements.txt
```

En Google Colab:
```python
!git clone https://github.com/TU_USUARIO/deep-learning-rnn.git
%cd deep-learning-rnn
!pip install -r requirements.txt
```

---

## Reproducibilidad — regla de oro

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

**Los splits los genera Yibby una sola vez.** Los demás los cargan desde `data/`:

```python
import pandas as pd
train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")
test_df  = pd.read_csv("data/test.csv")
```

---

## Cómo entrenar un modelo (10 líneas)

```python
from src.training import set_seed, train, evaluate
from src.metrics  import finalize_and_save
from src.preprocessing import build_pipeline

set_seed(42)

pipeline = build_pipeline({"data_dir": "data", "text_col": "review",
                            "label_col": "label", "vocab_size": 20000,
                            "max_len": 200, "batch_size": 64})

config = {"model_name": "lstm_v1", "owner": "Felipe", "track": "PP",
          "n_epochs": 30, "lr": 1e-3, "patience": 5,
          "checkpoint_path": "results/lstm_v1_best.pt",
          "class_weights": pipeline["class_weights"], "use_class_weights": True,
          "embedding_dim": 128, "hidden_size": 256, "dropout": 0.3}

metrics_dict = train(model, pipeline["train_loader"], pipeline["val_loader"], config)
y_true, y_pred = evaluate(model, pipeline["test_loader"], config["checkpoint_path"], config)
metrics_dict = finalize_and_save(y_true, y_pred, metrics_dict, show_plots=True)
```

---

## Formato JSON de métricas (estandarizado)

Todos los modelos guardan sus resultados en este formato para que la tabla comparativa final sea automática:

```json
{
  "model_name": "lstm_v1",
  "owner": "Felipe",
  "track": "PP",
  "config": { "embedding_dim": 128, "hidden_size": 256, "dropout": 0.3,
              "use_class_weights": true, "n_params": 542000 },
  "metrics": {
    "accuracy": 0.85, "precision_macro": 0.82, "recall_macro": 0.81, "f1_macro": 0.81,
    "f1_per_class": {"1": 0.70, "2": 0.65, "3": 0.78, "4": 0.84, "5": 0.90},
    "confusion_matrix": [[...]]
  },
  "training": {
    "epochs_run": 12, "best_epoch": 8, "training_time_seconds": 340,
    "loss_history": [...], "val_loss_history": [...],
    "acc_history": [...], "val_acc_history": [...]
  }
}
```

---

## Tabla comparativa final (script rápido)

```python
import json
from pathlib import Path
import pandas as pd

rows = []
for f in Path("results").glob("*_metrics.json"):
    d = json.load(open(f))
    rows.append({"Modelo": d["model_name"], "Owner": d["owner"],
                 "Accuracy": d["metrics"]["accuracy"],
                 "F1 macro": d["metrics"]["f1_macro"],
                 "Params": d["config"]["n_params"],
                 "Tiempo (s)": d["training"]["training_time_seconds"]})

df = pd.DataFrame(rows).sort_values("F1 macro", ascending=False)
print(df.to_markdown(index=False))
```

---

## Hitos

| Hito | Fecha | Responsable |
|---|---|---|
| ★ Hito 1: Cimientos listos | **Vie 8 mayo** | Todo el equipo |
| ★ Hito 2: Modelos congelados + métricas | **Jue 14 mayo** | Todo el equipo |
| ★ Hito 3: Borradores 1.0 completos | **Jue 21 mayo** | Todo el equipo |
| Entrega y presentación | **Lun 25 mayo** | Todo el equipo |

---

## Convenciones

- **Métricas:** siempre F1 macro y por clase. Nunca solo accuracy (dataset desbalanceado).
- **Test set:** solo se mira con `evaluate()`, al final, **una vez por modelo**.
- **Checkpoints:** `torch.save(model.state_dict(), path)` — siempre.
- **Antes de push:** ejecutar el notebook completo y dejar las salidas guardadas.
