# Plan de Avance Global — Proyecto Final Deep Learning

**Fecha de revision:** 25 mayo 2026
**Entrega y presentacion:** Lunes 25 de mayo de 2026
**Estado general:** URGENTE — modelos listos, documentos pendientes

---

## RESUMEN EJECUTIVO

### PROYECTO PRINCIPAL (PP)

| Componente | Responsable | Estado |
|---|---|---|
| EDA notebook 01 (dimensionalidad) | Felipe | LISTO |
| EDA notebook 02 (features/texto) | Yibby | LISTO |
| Infraestructura (src/) | Yibby + Daniel | LISTO |
| Datos (splits estratificados) | Yibby | LISTO |
| Modelo Clasico 1: LSTM | Felipe | LISTO — F1=0.3996 |
| Modelo Clasico 2: BiLSTM | Yibby | LISTO — F1=0.5749 |
| Modelo Combinado: BiLSTM+MultiHead | Daniel | LISTO — F1=0.6193 |
| Articulo IEEE (4-6 pag, PDF IEEE) | Felipe | BORRADOR ~85% en markdown — **FALTA PDF IEEE** |
| Reporte tecnico (PDF, sin limite pag) | Yibby | **NO INICIADO** |
| Presentacion (diapositivas) | Daniel | **NO INICIADA** |

### TAREA DE INVESTIGACION (TI)

| Componente | Responsable | Estado |
|---|---|---|
| Modelo Avanzado: BETO v3 fine-tuned | Sebastian | LISTO — F1=0.6565 (MEJOR del proyecto) |
| Notebook implementacion | Sebastian | LISTO — 06_transformer_sebas.ipynb |
| Informe de investigacion (max 5 pag) | Sebastian | BORRADOR en Word — **FALTA PDF final** |

---

## ESTADO POR PERSONA

### Felipe Reyes (PP)

**Completado:**
- [x] EDA dimensionalidad y variable objetivo (01_eda_felipe.ipynb)
- [x] Modelo LSTM entrenado (4 versiones, v3 definitiva con pipeline Yibby)
- [x] Metricas JSON, curvas, matriz de confusion exportadas
- [x] Correcciones de bugs en preprocessing.py y 02_eda_yibby.ipynb
- [x] Entrenamiento del BiLSTM de Yibby (ejecutado con outputs)
- [x] Borrador articulo IEEE (~85% en docs/articulo_ieee_borrador.md)
- [x] Mejoras al notebook 06 de Sebastian (encabezado, pre-tokenizacion, early stopping, variantes v2/v3)

**Pendiente:**
- [ ] **ARTICULO IEEE — completar seccion IV-D (combinado), Abstract, y generar PDF formato IEEE Conference**

**Metricas LSTM definitivo (v3):**
- F1 macro: 0.3996 | Accuracy: 0.6042
- F1 por clase: 1=0.686, 2=0.168, 3=0.359, 4=0.022, 5=0.764
- Parametros: 2,700,933 | Tiempo: 321.2s CPU | Mejor epoca: 19/24

---

### Yibby Gonzalez (PP)

**Completado:**
- [x] Pipeline de preprocesamiento (src/preprocessing.py)
- [x] Splits estratificados (train/val/test.csv, semilla 42)
- [x] EDA features y texto (02_eda_yibby.ipynb)
- [x] Modelo BiLSTM entrenado (2 versiones, v1 definitiva)
- [x] Metricas JSON, curvas, matriz de confusion exportadas

**Pendiente:**
- [ ] **REPORTE TECNICO DETALLADO (PDF, sin limite de paginas)**

**Metricas BiLSTM definitivo (v1, sin class weights):**
- F1 macro: 0.5749 | Accuracy: 0.6871
- F1 por clase: 1=0.716, 2=0.202, 3=0.564, 4=0.572, 5=0.821
- Parametros: 3,253,253 | Tiempo: 27,685.9s CPU | Mejor epoca: 4/8

---

### Daniel Ruiz (PP)

**Completado:**
- [x] Esqueleto de entrenamiento (src/training.py)
- [x] Modulo de metricas (src/metrics.py)
- [x] Modelo combinado: 3 variantes entrenadas en GPU
  - BiLSTM+Bahdanau: F1=0.5740
  - GRU+Bahdanau: F1=0.5785
  - BiLSTM+MultiHead: F1=0.6193 (DEFINITIVO)

**Pendiente:**
- [ ] **PRESENTACION DE SLIDES (.pptx + PDF)**

**Metricas BiLSTM+MultiHead definitivo:**
- F1 macro: 0.6193 | Accuracy: 0.6548
- F1 por clase: 1=0.756, 2=0.533, 3=0.549, 4=0.480, 5=0.778
- Parametros: 6,110,981 | Tiempo: 303.8s GPU | Mejor epoca: 11/16

---

### Sebastian Ruiz (TI)

**Completado:**
- [x] Notebook 06_transformer_sebas.ipynb (3 variantes entrenadas)
  - v1 (1 capa, balanced): F1=0.5926
  - v2 (3 capas, balanced, LR discriminativo): F1=0.6161
  - v3 (3 capas, sqrt balanced, LR discriminativo): F1=0.6565 (DEFINITIVO)
- [x] Metricas JSON y curvas integradas al repo
- [x] Informe de investigacion en Word (docs/Informe_Investigacion.docx)

**Pendiente:**
- [ ] **INFORME DE INVESTIGACION — convertir .docx a PDF final (max 5 paginas)**

**Metricas BETO v3 DEFINITIVO (MEJOR del proyecto):**
- F1 macro: 0.6565 | Accuracy: 0.7252
- F1 por clase: 1=0.777, 2=0.502, 3=0.595, 4=0.561, 5=0.848
- Parametros totales: 109,854,725 (entrenables: 21,858,053 = 19.9%)
- Tiempo: 1780s GPU local | Mejor epoca: 8/11 (early stopping)

---

## TABLAS COMPARATIVAS

### Tabla 1: Modelos definitivos (uno por rol)

| Modelo | Owner | Track | F1 macro | Accuracy | Prec macro | Recall macro | Params | Tiempo |
|---|---|---|---|---|---|---|---|---|
| LSTM v3 | Felipe | PP | 0.3996 | 0.6042 | 0.5639 | 0.4408 | 2,700,933 | 321s CPU |
| BiLSTM v1 | Yibby | PP | 0.5749 | 0.6871 | 0.5876 | 0.5795 | 3,253,253 | 27,686s CPU |
| BiLSTM+MultiHead | Daniel | PP | 0.6193 | 0.6548 | 0.6086 | 0.6454 | 6,110,981 | 304s GPU |
| BETO v3 fine-tuned | Sebastian | TI | **0.6565** | **0.7252** | 0.6544 | 0.6609 | 21,858,053 ent. | 1,780s GPU |

### Tabla 2: F1 por clase (modelos definitivos)

| Clase | LSTM v3 | BiLSTM v1 | BiLSTM+MultiHead | BETO v3 |
|---|---|---|---|---|
| 1 estrella | 0.686 | 0.716 | 0.756 | **0.777** |
| 2 estrellas | 0.168 | 0.202 | **0.533** | 0.502 |
| 3 estrellas | 0.359 | 0.564 | 0.549 | **0.595** |
| 4 estrellas | 0.022 | **0.572** | 0.480 | 0.561 |
| 5 estrellas | 0.764 | 0.821 | 0.778 | **0.848** |

### Tabla 3: Todas las variantes exploradas (ranking por F1 macro)

| # | Modelo | F1 macro | Accuracy | Params | Tiempo |
|---|---|---|---|---|---|
| 1 | BETO v3 (Sebastian) | **0.6565** | **0.7252** | 21.9M ent. | 1780s GPU |
| 2 | BiLSTM+MultiHead (Daniel) | 0.6193 | 0.6548 | 6.1M | 304s GPU |
| 3 | BETO v2 (Sebastian) | 0.6161 | 0.6864 | 21.9M ent. | 1566s GPU |
| 4 | BiLSTM v2 con weights (Yibby) | 0.5989 | 0.6453 | 3.3M | 1792s CPU |
| 5 | BETO v1 (Sebastian) | 0.5926 | 0.6706 | 7.7M ent. | 1036s GPU |
| 6 | GRU+Bahdanau (Daniel) | 0.5785 | 0.6416 | 4.6M | 119s GPU |
| 7 | BiLSTM v1 sin weights (Yibby) | 0.5749 | 0.6871 | 3.3M | 27686s CPU |
| 8 | BiLSTM+Bahdanau (Daniel) | 0.5740 | 0.6471 | 5.2M | 129s GPU |
| 9 | LSTM v3 (Felipe) | 0.3996 | 0.6042 | 2.7M | 321s CPU |

---

## INVENTARIO DE ARCHIVOS

### PROYECTO PRINCIPAL (PP)

**Data:**

| Archivo | Estado |
|---|---|
| data/Big_AHR.csv | LISTO — dataset completo (18,172 filas, 7 columnas) |
| data/train.csv | LISTO — 12,720 muestras |
| data/val.csv | LISTO — 2,726 muestras |
| data/test.csv | LISTO — 2,726 muestras |

**Codigo fuente (src/):**

| Archivo | Autor | Estado |
|---|---|---|
| src/preprocessing.py | Yibby | LISTO |
| src/training.py | Daniel | LISTO |
| src/metrics.py | Daniel | LISTO |
| src/__init__.py | — | LISTO |

**Notebooks (PP):**

| Archivo | Autor | Estado |
|---|---|---|
| notebooks/01_eda_felipe.ipynb | Felipe | LISTO — ejecutado con outputs |
| notebooks/02_eda_yibby.ipynb | Yibby/Felipe | LISTO — ejecutado con outputs |
| notebooks/03_lstm_felipe.ipynb | Felipe | LISTO — 4 versiones, v3 definitiva |
| notebooks/04_bilstm_yibby.ipynb | Yibby/Felipe | LISTO — 2 versiones, v1 definitiva |
| notebooks/05_combined_daniel.ipynb | Daniel | LISTO — 3 variantes, BiLSTM+MultiHead definitivo |

**Resultados definitivos (PP):**

| Archivo | Modelo |
|---|---|
| results/lstm_metrics.json | LSTM v3 (Felipe) — F1=0.3996 |
| results/bilstm_v1_metrics.json | BiLSTM v1 (Yibby) — F1=0.5749 |
| results/bilstm_multihead_metrics.json | BiLSTM+MultiHead (Daniel) — F1=0.6193 |

**Resultados exploratorios (PP):**

| Archivo | Modelo |
|---|---|
| results/bilstm_metrics.json | Copia de bilstm_v1 (corregido) |
| results/bilstm_v2_metrics.json | BiLSTM v2 con class weights — F1=0.5989 |
| results/bilstm_bahdanau_metrics.json | BiLSTM+Bahdanau (Daniel) — F1=0.5740 |
| results/gru_bahdanau_metrics.json | GRU+Bahdanau (Daniel) — F1=0.5785 |
| results/lstm_v2_metrics.json | LSTM v2 historico |
| results/lstm_v3_yibby_metrics.json | LSTM v3 copia |
| results/lstm_v4_sqrt_metrics.json | LSTM v4 historico |

**Checkpoints (.pt) — PP:**

| Archivo | Estado |
|---|---|
| results/lstm_best.pt | LISTO (10.3MB) |
| results/bilstm_best.pt | LISTO (13MB) |
| results/bilstm_v1_best.pt | LISTO (13MB) |
| results/bilstm_bahdanau_best.pt | LISTO (20.7MB) |
| results/bilstm_multihead_best.pt | FALTA — generado en GPU, no copiado |
| results/gru_bahdanau_best.pt | FALTA — generado en GPU, no copiado |

**Figuras EDA (PP):**

| Archivo | Estado |
|---|---|
| figures/eda_distribucion_clases.png | LISTO |
| figures/eda_histograma_longitudes.png | LISTO |
| figures/eda_boxplot_longitud_clase.png | LISTO |
| figures/eda_zipf_cobertura.png | LISTO |
| figures/eda_top50_words_global.png | LISTO |
| figures/eda_top30_por_clase.png | LISTO |
| figures/eda_wordclouds_por_clase.png | LISTO |
| figures/eda_class_distribution_weights.png | LISTO |

**Figuras modelos (PP):**

| Archivo | Estado |
|---|---|
| figures/lstm_curves.png | LISTO |
| figures/lstm_confusion_matrix.png | LISTO |
| figures/lstm_v2_curves.png | LISTO |
| figures/lstm_v2_confusion_matrix.png | LISTO |
| figures/lstm_v3_yibby_curves.png | LISTO |
| figures/lstm_v3_yibby_confusion_matrix.png | LISTO |
| figures/lstm_v4_sqrt_curves.png | LISTO |
| figures/lstm_v4_sqrt_confusion_matrix.png | LISTO |
| figures/bilstm_v1_curves.png | LISTO |
| figures/bilstm_v1_confusion_matrix.png | LISTO |
| figures/bilstm_v2_curves.png | LISTO |
| figures/bilstm_v2_confusion_matrix.png | LISTO |
| figures/bilstm_analisis_final.png | LISTO |
| figures/bilstm_bahdanau_curves.png | LISTO |
| figures/bilstm_bahdanau_confusion_matrix.png | LISTO |
| figures/gru_bahdanau_curves.png | LISTO |
| figures/gru_bahdanau_confusion_matrix.png | LISTO |
| figures/bilstm_multihead_curves.png | LISTO |
| figures/bilstm_multihead_confusion_matrix.png | LISTO |
| figures/combined_f1_por_clase.png | LISTO |

**Documentos (PP):**

| Documento | Estado | Responsable |
|---|---|---|
| docs/articulo_ieee_borrador.md | BORRADOR ~85% | Felipe |
| Articulo IEEE final (PDF IEEE Conference) | **FALTA** | Felipe |
| Reporte tecnico detallado (PDF) | **FALTA — NO INICIADO** | Yibby |
| Presentacion (.pptx + PDF) | **FALTA — NO INICIADA** | Daniel |

---

### TAREA DE INVESTIGACION (TI)

**Notebook (TI):**

| Archivo | Autor | Estado |
|---|---|---|
| notebooks/06_transformer_sebas.ipynb | Sebastian | LISTO — 3 variantes, v3 definitiva |

**Resultados (TI):**

| Archivo | Modelo |
|---|---|
| results/transformer_metrics.json | BETO v3 (Sebastian) — F1=0.6565 |

**Checkpoints (TI):**

| Archivo | Estado |
|---|---|
| results/transformer_best.pt | FALTA — generado en GPU local (~440MB) |

**Figuras (TI):**

| Archivo | Estado |
|---|---|
| figures/transformer_curves.png | LISTO (v1 baseline) |
| figures/transformer_confusion.png | LISTO (v1 normalizada) |
| figures/transformer_confusion_absolutos.png | LISTO (v1 absolutos) |

**Nota:** Las figuras del transformer corresponden a v1 (baseline). Faltan las curvas y confusion matrix de v3 (modelo definitivo).

**Documentos (TI):**

| Documento | Estado | Responsable |
|---|---|---|
| docs/Informe_Investigacion.docx | BORRADOR en Word | Sebastian |
| Informe de investigacion final (PDF, max 5 pag) | **FALTA — exportar a PDF** | Sebastian |

---

## LIMITACIONES CONOCIDAS

1. **Data leakage entre splits:** Train/Val comparten 868 resenas, Train/Test comparten 902, Val/Test comparten 249. Duplicados exactos dentro de train: 2,674 (21%). Documentar como limitacion en articulo y reporte.

2. **MAX_LEN diferente entre modelos:** LSTM=150, BiLSTM/Combinado=200, Transformer=128 (subword). No se puede corregir sin reentrenar.

3. **Figuras transformer desactualizadas:** Solo hay figuras de v1, no de v3 (definitivo).

---

## CHECKLIST FINAL DE ENTREGA

### Proyecto Principal (PP) — 4 entregables

- [ ] Articulo IEEE en PDF formato Conference (4-6 paginas) — Felipe
- [ ] Reporte tecnico detallado en PDF — Yibby
- [ ] Presentacion en diapositivas — Daniel
- [x] Notebooks ejecutados con outputs (01 a 05)

### Tarea de Investigacion (TI) — 2 entregables

- [ ] Informe de investigacion en PDF (max 5 paginas) — Sebastian
- [x] Notebook ejecutado con outputs (06)

### Comunes

- [x] Repositorio con codigo y estructura limpia
- [x] README.md actualizado
- [x] Metricas JSON de los 4 modelos definitivos
- [x] Figuras de EDA, curvas y confusion matrices
