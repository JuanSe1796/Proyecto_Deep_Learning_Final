# Plan de Avance Global — Reunion 25 mayo 2026

**Fecha de revision:** 25 mayo 2026 (noche) — ACTUALIZADO CON AVANCES DEL DIA
**Entrega:** Lunes 26 mayo 2026
**Estado general:** URGENTE — modelos mejorados, faltan documentos

---

## RESUMEN EJECUTIVO

| Componente | Estado | Bloqueante para entrega |
|---|---|---|
| EDA (2 notebooks) | LISTO | No |
| Infraestructura (src/) | LISTO | No |
| Datos (splits) | LISTO | No |
| LSTM (Felipe) | LISTO - F1=0.3996 | No |
| BiLSTM (Yibby) | LISTO - F1=0.5568 | No |
| Modelo Combinado (Daniel) | **LISTO** - 3 variantes, mejor: BiLSTM+MultiHead F1=**0.6193** | No |
| Transformer (Sebastian) | **LISTO** - 3 variantes validadas, v3 definitiva F1=**0.6565** (MEJOR del proyecto) | No |
| Articulo IEEE (Felipe) | BORRADOR ~85% en docs/articulo_ieee_borrador.md | **SI — falta PDF final** |
| Reporte tecnico (Yibby) | **NO INICIADO** | **SI** |
| Presentacion (Daniel) | **NO INICIADO** | **SI** |
| Informe investigacion (Sebastian) | BORRADOR en Word (`docs/Informe_Investigacion.docx`) | **SI — falta PDF final** |

### CAMBIOS DEL 25 DE MAYO (sesion con Claude)

**Notebook 06_transformer_sebas.ipynb — cambios realizados:**

1. **Encabezado profesional** agregado (autor, proyecto, universidad, estrategia) — estilo consistente con el de Felipe
2. **Numeracion y markdown** en todas las secciones (1-13)
3. **Pre-tokenizacion** del dataset (optimizacion: tokeniza una sola vez en `__init__` en vez de en cada `__getitem__` por epoca)
4. **Early stopping** implementado (paciencia=3, restaura mejor checkpoint automaticamente)
5. **N_EPOCHS** subio de 5 a 15 (el early stopping corta antes)
6. **Celdas de Colab** comentadas (Setup Colab y google.colab.files) para que no falle fuera de Colab
7. **Tres variantes** de entrenamiento agregadas para experimentacion:

| Variante | Capas descongeladas | Class weights | LR strategy | Estado |
|---|---|---|---|---|
| v1 (baseline) | layer.11 (7%) | balanced | uniforme 2e-5 | **EJECUTADO** — F1=0.5926 |
| v2 | layers 9,10,11 (~20%) | balanced | discriminativo (2e-6/1e-5/2e-5) | **EJECUTADO** — F1=0.6161, Acc=0.6864, best epoch 6, 1566s |
| v3 | layers 9,10,11 (~20%) | sqrt(balanced) | discriminativo (2e-6/1e-5/2e-5) | **EJECUTADO — DEFINITIVO** — F1=0.6565, Acc=0.7252, best epoch 8, 1780s |

8. **Seccion 13**: Comparacion final automatica que selecciona la mejor version y guarda el JSON/checkpoint definitivo

**Otros archivos creados:**
- `/freyesTemp/setup_transformer_gpu.md` — Guia paso a paso Miniconda + GPU para correr el notebook
- Copiado tambien a `Proyecto_Deep_Learning_Final/freyesTemp/`

---

### VALIDACION COMPLETADA (25 mayo — sesion nocturna)

**v2 y v3 del transformer ejecutadas y validadas.** Resultados:
- v1: F1=0.5926 | v2: F1=0.6161 | **v3: F1=0.6565 (GANADOR)**
- La seccion 13 del notebook selecciono correctamente v3 como ganador
- `transformer_metrics.json` actualizado con metricas de v3
- `transformer_best.pt` actualizado con checkpoint de v3
- BETO v3 es ahora el **MEJOR modelo de todo el proyecto** (supera a BiLSTM+MultiHead por 3.7 puntos F1)

**Archivos nuevos detectados:**
- `docs/Informe_Investigacion.docx` — Informe de investigacion de Sebastian (borrador en Word)
- `figures/transformer_confusion_absolutos.png` — Matriz de confusion con valores absolutos

---

## ESTADO POR PERSONA

### Felipe Reyes

**Completado:**
- [x] EDA dimensionalidad y variable objetivo (01_eda_felipe.ipynb) — 12/12 items
- [x] Modelo LSTM entrenado (4 versiones, v3 definitiva con pipeline Yibby)
- [x] Metricas JSON, curvas, matriz de confusion exportadas
- [x] Correccion de bugs en preprocessing.py y 02_eda_yibby.ipynb
- [x] Correccion del notebook EDA de Yibby (ejecutado con outputs)
- [x] Entrenamiento del BiLSTM de Yibby (ejecutado con outputs)
- [x] Borrador articulo IEEE (~85% listo en docs/articulo_ieee_borrador.md)
- [x] Mejoras al notebook 06 de Sebastian (encabezado, numeracion, pre-tokenizacion, early stopping, variantes v2/v3)
- [x] Guia de instalacion Miniconda+GPU (freyesTemp/setup_transformer_gpu.md)

**Pendiente:**
- [ ] **ARTICULO IEEE — completar y pasar a PDF formato IEEE Conference**
- [x] Validacion de v2 y v3 del transformer completada (v3 ganador, F1=0.6565)

**Metricas LSTM definitivo (v3):**
- F1 macro: 0.3996 | Accuracy: 0.6042
- F1 por clase: 1=0.686, 2=0.168, 3=0.359, 4=0.022, 5=0.764
- Parametros: 2,700,933 | Tiempo: 321.2s CPU | Mejor epoca: 19

---

### Yibby Gonzalez

**Completado:**
- [x] Pipeline de preprocesamiento (src/preprocessing.py)
- [x] Splits estratificados (train/val/test.csv)
- [x] EDA features y texto (02_eda_yibby.ipynb)
- [x] Modelo BiLSTM entrenado (2 versiones, v1 definitiva)
- [x] Metricas JSON, curvas, matriz de confusion exportadas

**Pendiente:**
- [ ] **REPORTE TECNICO DETALLADO** — responsabilidad de Yibby

**Metricas BiLSTM definitivo (v1):**
- F1 macro: 0.5749 | Accuracy: 0.6871
- F1 por clase: 1=0.716, 2=0.202, 3=0.564, 4=0.572, 5=0.821
- Parametros: 3,253,253 | Tiempo: 27,685.9s CPU | Mejor epoca: 4

---

### Daniel Ruiz

**Completado:**
- [x] Esqueleto de entrenamiento (src/training.py)
- [x] Modulo de metricas (src/metrics.py)
- [x] **MODELO COMBINADO — COMPLETO Y EJECUTADO** (notebook 05)

**Pendiente:**
- [ ] **PRESENTACION DE SLIDES**
- [ ] Copiar checkpoints (.pt) de los 3 modelos al repositorio

**Modelo definitivo: BiLSTM+MultiHead (MEJOR del proyecto):**
- F1 macro: 0.6193 | Accuracy: 0.6548
- F1 por clase: 1=0.756, 2=0.533, 3=0.549, 4=0.480, 5=0.778
- Parametros: 6,110,981 | Tiempo: 303.8s GPU | Mejor epoca: 11

---

### Sebastian Ruiz

**Completado:**
- [x] Notebook 06_transformer_sebas.ipynb (reorganizado y mejorado)
- [x] BETO v1 fine-tuned con early stopping (7 epocas, best=4)
- [x] BETO v2 fine-tuned (3 capas, balanced, lr discrim.) — F1=0.6161
- [x] BETO v3 fine-tuned (3 capas, sqrt, lr discrim.) — **F1=0.6565 DEFINITIVO**
- [x] Metricas JSON integradas al repo (actualizadas con v3)
- [x] Curvas y matrices de confusion integradas
- [x] Informe de investigacion en Word (`docs/Informe_Investigacion.docx`)

**Pendiente:**
- [ ] **INFORME DE INVESTIGACION — convertir a PDF final** (max 5 paginas)
- [ ] Copiar checkpoints transformer (.pt) al repositorio

**Metricas BETO v3 DEFINITIVO (MEJOR del proyecto):**
- F1 macro: **0.6565** | Accuracy: **0.7252**
- F1 por clase: 1=0.777, 2=0.502, 3=0.595, 4=0.561, 5=0.848
- Parametros totales: 109,854,725 (entrenables: 21,858,053 = 19.9%)
- Tiempo: 1780s en GPU local | Mejor epoca: 8 (early stopping en 11)

**Historico de variantes:**

| Variante | F1 macro | Accuracy | Capas | Weights | Best epoch | Tiempo |
|---|---|---|---|---|---|---|
| v1 | 0.5926 | 0.6706 | layer.11 (7%) | balanced | 4 | 1036s |
| v2 | 0.6161 | 0.6864 | layers 9-11 (20%) | balanced | 6 | 1566s |
| **v3** | **0.6565** | **0.7252** | layers 9-11 (20%) | sqrt(balanced) | 8 | 1780s |

---

## TABLA COMPARATIVA (actualizada al 25 mayo — COMPLETA)

| # | Modelo | F1 macro | Accuracy | F1 2est | F1 3est | Params | Tiempo |
|---|---|---|---|---|---|---|---|
| 1 | **BETO v3 (Sebastian)** | **0.6565** | **0.7252** | 0.502 | 0.595 | 21.9M ent. | 1780s GPU |
| 2 | **BiLSTM+MultiHead (Daniel)** | 0.6193 | 0.6548 | 0.533 | 0.549 | 6.1M | 303.8s GPU |
| 3 | BETO v2 (Sebastian) | 0.6161 | 0.6864 | 0.481 | 0.517 | 21.9M ent. | 1566s GPU |
| 4 | BETO v1 (Sebastian) | 0.5926 | 0.6706 | 0.419 | 0.513 | 7.7M ent. | 1036s GPU |
| 5 | GRU+Bahdanau (Daniel) | 0.5785 | 0.6416 | 0.436 | 0.457 | 4.6M | 119.4s GPU |
| 6 | BiLSTM v1 (Yibby) | 0.5749 | 0.6871 | 0.202 | 0.564 | 3.3M | 27,685.9s CPU |
| 7 | BiLSTM+Bahdanau (Daniel) | 0.5740 | 0.6471 | 0.433 | 0.479 | 5.2M | 129.3s GPU |
| 8 | LSTM v3 (Felipe) | 0.3996 | 0.6042 | 0.168 | 0.359 | 2.7M | 321.2s CPU |

---

## CHECKLIST FINAL

- [x] Repositorio Git con todo el codigo, notebooks ejecutados
- [x] README.md actualizado con inventario completo
- [x] Metricas JSON de los 4 modelos definitivos
- [x] Figuras de EDA, curvas y confusion matrices
- [x] Notebook transformer mejorado (early stopping, variantes, pre-tokenizacion)
- [x] Validar v2/v3 del transformer — COMPLETADO (v3 ganador, F1=0.6565)
- [ ] Articulo IEEE en PDF — Felipe
- [ ] Reporte tecnico en PDF — Yibby
- [ ] Presentacion en PDF + .pptx — Daniel
- [ ] Informe de investigacion en PDF — Sebastian (borrador Word listo, falta PDF)
- [ ] Checkpoints de modelos combinados (.pt) — Daniel
- [ ] Checkpoints de transformer (.pt) — Sebastian
