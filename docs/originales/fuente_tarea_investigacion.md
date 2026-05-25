# FUENTE DE VERDAD — Tarea de Investigacion (TI)

> Este documento contiene TODA la informacion tecnica de la Tarea de Investigacion.
> Sirve como referencia unica para redactar el Informe de Investigacion (max 5 paginas).
> No es necesario consultar notebooks ni JSONs — todo esta aqui.

---

## 1. INFORMACION GENERAL

**Asignatura:** Aprendizaje Profundo — Maestria en IA
**Universidad:** Pontificia Universidad Javeriana, Bogota, 2026
**Profesor:** Ing. Julio Omar Palacio Nino, M.Sc.
**Responsable:** Sebastian Ruiz
**Entrega:** Lunes 25 de mayo de 2026

### 1.1 Que es la Tarea de Investigacion

Segun el profesor (Proyecto 2026-3.pdf):

> "Esta fase es una tarea de investigacion tecnica que complementa el proyecto principal. Corresponde a la implementacion del 'nuevo modelo' solicitado en el proyecto, el cual debe demostrar una mayor profundidad tecnica. El objetivo es realizar un analisis experimental comparativo que enfrente los modelos clasicos contra arquitecturas mas avanzadas o tecnicas modernas de Deep Learning."

### 1.2 Tarea especifica (Dataset RNN)

> "Implementar un modelo usando Fine-Tuning de un Transformer. Detalle: Deberan usar un modelo de lenguaje pre-entrenado (ej. BETO, RoBERTa) y realizar fine-tuning de (al menos) la ultima capa del transformer para la tarea de clasificacion."

### 1.3 Entregables (exactamente 2)

1. **Notebook** — implementacion, entrenamiento y evaluacion del modelo avanzado.
2. **Informe de Investigacion (PDF, max 5 paginas)** con estructura obligatoria:
   - Introduccion
   - Metodologia
   - Resultados (tabla comparativa + curvas loss/accuracy)
   - Analisis y Conclusiones

### 1.4 Estructura obligatoria del informe

| Seccion | Contenido requerido |
|---|---|
| Introduccion | Breve descripcion del objetivo (ej. "Evaluar el impacto de Fine-Tuning...") |
| Metodologia | Arquitectura avanzada, modelo base, capas descongeladas, hiperparametros |
| Resultados | Tabla comparativa avanzado vs 2 clasicos + curvas loss/accuracy del avanzado |
| Analisis y Conclusiones | Superacion o no, por que, dificultad de implementacion, costo computacional, ajuste de hiperparametros |

---

## 2. DATASET Y SPLITS

**Dataset:** Andalusian Hotels Reviews (Unbalanced)
**URL:** https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced
**Tarea:** Clasificacion de sentimiento (5 clases: 1-5 estrellas)
**Total:** 18,172 resenas en espanol

**Splits (identicos a los del Proyecto Principal):**
- Train: 12,720 (70%)
- Validacion: 2,726 (15%)
- Test: 2,726 (15%)
- Estratificados, semilla 42

**Distribucion de clases:**

| Clase | Rating | Muestras (train) | % |
|---|---|---|---|
| 0 | 1 estrella | 1,174 | 9.2% |
| 1 | 2 estrellas | 696 | 5.5% |
| 2 | 3 estrellas | 1,592 | 12.5% |
| 3 | 4 estrellas | 2,955 | 23.2% |
| 4 | 5 estrellas | 6,303 | 49.6% |

**Desbalance:** ratio 9.1:1 entre clase mayoritaria (5★) y minoritaria (2★).

**Comparabilidad confirmada:** Sebastian uso exactamente los mismos splits del repositorio (data/train.csv, val.csv, test.csv), clonando el repo en su entorno GPU. Las metricas son directamente comparables con las de los modelos clasicos.

---

## 3. MODELO AVANZADO: BETO FINE-TUNED

### 3.1 Modelo base

- **Nombre:** BETO cased
- **HuggingFace ID:** `dccuchile/bert-base-spanish-wwm-cased`
- **Tipo:** BERT pre-entrenado en espanol con Whole Word Masking
- **Arquitectura:** 12 capas encoder, 768 hidden size, 12 attention heads
- **Parametros totales:** 109,854,725
- **Vocabulario:** ~31,000 subword tokens (WordPiece)

### 3.2 Estrategia de Fine-Tuning

Se descongelan las ultimas capas del encoder y se entrenan junto con una cabeza de clasificacion nueva:

```
[CONGELADO] encoder.layer.0 a encoder.layer.8 (75.4% de params, preservan conocimiento general)
[ENTRENABLE] encoder.layer.9   (lr=2e-6, adaptacion lenta)
[ENTRENABLE] encoder.layer.10  (lr=1e-5, adaptacion media)
[ENTRENABLE] encoder.layer.11  (lr=2e-5, adaptacion rapida)
[ENTRENABLE] pooler             (lr=2e-5)
[ENTRENABLE] classifier         (lr=2e-5, capa nueva desde cero)
```

**Parametros entrenables:** 21,858,053 / 109,854,725 = **19.9%**

### 3.3 Tres versiones exploradas

| Version | Capas descongeladas | Class Weights | LR | F1 macro | Accuracy | Tiempo |
|---|---|---|---|---|---|---|
| v1 | layer.11 (7%) | balanced | uniforme 2e-5 | 0.5926 | 0.6706 | 1,036s |
| v2 | layers 9,10,11 (20%) | balanced | discriminativo | 0.6161 | 0.6864 | 1,566s |
| **v3** | **layers 9,10,11 (20%)** | **sqrt(balanced)** | **discriminativo** | **0.6565** | **0.7252** | **1,780s** |

### 3.4 Innovaciones clave entre versiones

**v1 -> v2 (+2.4 pp F1):**
- Mas capas descongeladas: 3 capas encoder vs 1
- LR discriminativo: capas inferiores aprenden mas lento, preservando representaciones generales

**v2 -> v3 (+4.0 pp F1):**
- sqrt(balanced) en vez de balanced: suaviza los pesos de clase
  - balanced: [2.167, 3.655, 1.598, 0.861, 0.404]
  - sqrt(balanced): [1.472, 1.912, 1.264, 0.928, 0.635]
- Evita sobrecompensacion de clases raras que degradaba accuracy

### 3.5 Hiperparametros definitivos (v3)

| Parametro | Valor |
|---|---|
| base_model | dccuchile/bert-base-spanish-wwm-cased |
| capas descongeladas | encoder.layer.9, 10, 11 + pooler + classifier |
| params entrenables | 21,858,053 (19.9% del total) |
| params totales | 109,854,725 |
| max_length | 128 subword tokens |
| batch_size | 32 |
| max_epochs | 15 |
| early_stopping_patience | 3 |
| optimizer | AdamW |
| weight_decay | 0.01 |
| lr encoder.layer.9 | 2e-6 |
| lr encoder.layer.10 | 1e-5 |
| lr encoder.layer.11 + pooler + classifier | 2e-5 |
| class_weights | sqrt(balanced) = [1.472, 1.912, 1.264, 0.928, 0.635] |
| warmup | 10% de steps totales (linear) |
| gradient_clipping | max_norm=1.0 |
| tokenizer | BETO cased tokenizer (subword WordPiece) |

---

## 4. METRICAS DEL MODELO AVANZADO (v3 — DEFINITIVO)

### 4.1 Metricas principales

| Metrica | Valor |
|---|---|
| **F1 macro** | **0.6565** |
| **Accuracy** | **0.7252** |
| Precision macro | 0.6544 |
| Recall macro | 0.6609 |
| Parametros entrenables | 21,858,053 |
| Parametros totales | 109,854,725 |
| Tiempo entrenamiento | 1,780 segundos (GPU) |
| Mejor epoca | 8 / 11 (early stopping en epoca 11) |

### 4.2 F1 por clase

| Clase | F1 |
|---|---|
| 1 estrella | 0.777 |
| 2 estrellas | 0.502 |
| 3 estrellas | 0.595 |
| 4 estrellas | 0.561 |
| 5 estrellas | 0.848 |

### 4.3 Matriz de confusion

```
Pred:       1★    2★    3★    4★    5★
Real 1★  [ 192,   50,    9,    1,    0]
Real 2★  [  33,   82,   30,    4,    0]
Real 3★  [  15,   40,  204,   64,   18]
Real 4★  [   1,    5,   80,  339,  208]
Real 5★  [   1,    1,   22,  167, 1160]
```

### 4.4 Historiales de entrenamiento

**Loss (11 epocas):**
- Train: [1.3339, 0.9136, 0.8231, 0.7808, 0.7305, 0.6908, 0.6468, 0.6098, 0.5647, 0.5365, 0.5112]
- Val: [0.9942, 0.8671, 0.8314, 0.8106, 0.7939, 0.7804, 0.7753, 0.7611, 0.7661, 0.7682, 0.7759]

**Accuracy (11 epocas):**
- Train: [0.4891, 0.6481, 0.6808, 0.7003, 0.7153, 0.7257, 0.7535, 0.7671, 0.7838, 0.7968, 0.8058]
- Val: [0.6258, 0.6834, 0.6922, 0.7047, 0.7054, 0.7146, 0.7142, 0.7267, 0.7388, 0.7392, 0.7454]

**Observacion:** Val loss alcanza minimo en epoca 8 (0.7611) y luego sube ligeramente (0.7661, 0.7682, 0.7759). Early stopping corta correctamente en epoca 11 (3 epocas sin mejora). El modelo convergio adecuadamente.

---

## 5. MODELOS CLASICOS (referencia para tabla comparativa)

### 5.1 Modelo Clasico 1: LSTM (Felipe)

**Arquitectura:**
```
Embedding(20002, 128) -> LSTM(128, 1 capa) -> Dense(128->64, ReLU) -> Dropout(0.3) -> Dense(64->5)
```

**Hiperparametros clave:** lr=3e-4, MAX_LEN=150, sin class weights, patience=5

| Metrica | Valor |
|---|---|
| F1 macro | 0.3996 |
| Accuracy | 0.6042 |
| Precision macro | 0.5639 |
| Recall macro | 0.4408 |
| Parametros | 2,700,933 |
| Tiempo | 321.2s (CPU) |
| Mejor epoca | 19 / 24 |

**F1 por clase:** 1★=0.686, 2★=0.168, 3★=0.359, 4★=0.022, 5★=0.764

### 5.2 Modelo Clasico 2: BiLSTM (Yibby)

**Arquitectura:**
```
Embedding(20002, 128) -> BiLSTM(128/dir, 2 capas) -> Global Max Pool -> Dropout(0.4) -> Dense(256->128, ReLU) -> Dense(128->5)
```

**Hiperparametros clave:** lr=1e-3, MAX_LEN=200, sin class weights, patience=4

| Metrica | Valor |
|---|---|
| F1 macro | 0.5749 |
| Accuracy | 0.6871 |
| Precision macro | 0.5876 |
| Recall macro | 0.5795 |
| Parametros | 3,253,253 |
| Tiempo | 27,685.9s (CPU) |
| Mejor epoca | 4 / 8 |

**F1 por clase:** 1★=0.716, 2★=0.202, 3★=0.564, 4★=0.572, 5★=0.821

### 5.3 Mejor modelo clasico

**BiLSTM de Yibby** (F1 macro = 0.5749, accuracy = 0.6871)

---

## 6. TABLA COMPARATIVA OBLIGATORIA

### 6.1 Metricas principales

| Modelo | F1 macro | Accuracy | Precision macro | Recall macro | Params | Tiempo |
|---|---|---|---|---|---|---|
| LSTM (Felipe) | 0.3996 | 0.6042 | 0.5639 | 0.4408 | 2,700,933 | 321s CPU |
| BiLSTM (Yibby) | 0.5749 | 0.6871 | 0.5876 | 0.5795 | 3,253,253 | 27,686s CPU |
| **BETO v3 (Sebastian)** | **0.6565** | **0.7252** | **0.6544** | **0.6609** | **21,858,053 ent.** | **1,780s GPU** |

### 6.2 F1 por clase

| Clase | LSTM | BiLSTM | BETO v3 | Mejor |
|---|---|---|---|---|
| 1 estrella | 0.686 | 0.716 | **0.777** | BETO |
| 2 estrellas | 0.168 | 0.202 | **0.502** | BETO |
| 3 estrellas | 0.359 | 0.564 | **0.595** | BETO |
| 4 estrellas | 0.022 | **0.572** | 0.561 | BiLSTM |
| 5 estrellas | 0.764 | 0.821 | **0.848** | BETO |

### 6.3 Mejoras de BETO sobre el mejor clasico (BiLSTM)

| Metrica | BiLSTM | BETO v3 | Delta |
|---|---|---|---|
| F1 macro | 0.5749 | 0.6565 | **+8.2 pp** |
| Accuracy | 0.6871 | 0.7252 | **+3.8 pp** |
| Precision macro | 0.5876 | 0.6544 | +6.7 pp |
| Recall macro | 0.5795 | 0.6609 | +8.1 pp |

BETO supera al mejor clasico en **4 de 5 clases**. La unica clase donde BiLSTM mantiene ventaja es 4★ (0.572 vs 0.561, diferencia de solo 1.1 pp).

---

## 7. ANALISIS Y CONCLUSIONES

### 7.1 El modelo avanzado supero a los clasicos? SI

- BETO v3 supera al mejor clasico (BiLSTM) por **+8.2 puntos de F1 macro**
- Supera al LSTM por **+25.7 puntos de F1 macro**
- Supera en 4 de 5 clases individuales
- Mejora especialmente en clases minoritarias: clase 2★ sube de 0.202 a 0.502 (+30 pp)

### 7.2 Por que supera?

1. **Embeddings contextuales pre-entrenados:** BETO fue pre-entrenado en un corpus masivo de espanol. Conoce relaciones semanticas complejas (sinonimos, antonimos, dobles negaciones) que los modelos clasicos deben aprender desde cero con solo 12,720 ejemplos.

2. **Representaciones subword:** El tokenizer WordPiece de BETO maneja palabras raras o mal escritas descomponiendolas en subpalabras conocidas, mientras que el vocabulario de 20K palabras del pipeline clasico las marca como UNK.

3. **Atencion multi-cabeza nativa:** Los 12 attention heads de BERT capturan dependencias a larga distancia de forma mas efectiva que las LSTMs.

4. **Transfer learning:** El conocimiento general del idioma (gramatica, semantica) esta codificado en las capas congeladas. Solo se adaptan las capas superiores al dominio hotelero.

### 7.3 Dificultad de implementacion

| Aspecto | Dificultad | Detalle |
|---|---|---|
| Instalacion | Media | Requiere transformers, accelerate, torch con CUDA. Versiones deben ser compatibles |
| GPU obligatoria | Alta restriccion | Imposible entrenar en CPU razonablemente (~10x mas lento) |
| Overfitting | Alta | Transformers hacen overfit muy rapido. Early stopping con paciencia baja (3) es esencial |
| Hiperparametros | Alta | LR discriminativo y tipo de class weights requirieron experimentacion. 3 versiones para encontrar configuracion optima |
| Debugging | Media | Errores de CUDA, memory overflow con batch_size grande, problemas de tokenizacion |
| Reproducibilidad | Media | Semilla no garantiza 100% reproducibilidad en GPU por operaciones no-deterministicas |

### 7.4 Costo computacional

| Recurso | LSTM | BiLSTM | BETO v3 | Ratio BETO/BiLSTM |
|---|---|---|---|---|
| Parametros entrenables | 2.7M | 3.3M | 21.9M | 6.6x |
| Tiempo entrenamiento | 321s CPU | 27,686s CPU | 1,780s GPU | — |
| GPU requerida | No | No | **SI** | — |
| Memoria GPU estimada | — | — | ~6-8 GB | — |
| Epocas hasta convergencia | 19 | 4 | 8 | — |

**Nota sobre tiempos:** El BiLSTM tardo 27,686s porque se entreno en CPU. En GPU tardaria ~300-500s (similar al modelo combinado de Daniel con arquitectura comparable). BETO en GPU tarda 1,780s — mas lento que RNNs equivalentes en GPU pero aun razonable.

### 7.5 Ajuste de hiperparametros

| Parametro | Lo que se probo | Lo que funciono mejor | Por que |
|---|---|---|---|
| Capas descongeladas | 1 capa vs 3 capas | 3 capas (layers 9,10,11) | Mas capacidad de adaptacion al dominio sin destruir representaciones generales |
| Learning rate | Uniforme 2e-5 vs discriminativo | Discriminativo (2e-6/1e-5/2e-5) | Capas inferiores preservan conocimiento general; superiores se adaptan mas rapido |
| Class weights | balanced vs sqrt(balanced) | sqrt(balanced) | balanced sobrecompensa clases raras, degradando accuracy global |
| Max length | 128 | 128 | Suficiente para >95% de resenas en subword tokens |
| Batch size | 32 | 32 | Mayor batch causaba OOM; menor era mas lento sin beneficio |
| Early stopping patience | 3 | 3 | Transformers overfit rapido; paciencia baja previene degradacion |

### 7.6 Limitaciones del estudio

1. **Data leakage en splits:** 868 resenas compartidas train/val, 902 train/test. Las metricas pueden estar ligeramente infladas.
2. **MAX_LEN 128 vs 200:** BETO trunca a 128 subword tokens (~100-150 palabras). Resenas largas pierden informacion. Los modelos clasicos usan 150-200 word tokens.
3. **Solo se probo BETO cased:** no se compararon otros transformers (RoBERTa-BNE, MarIA, BERTIN) por restricciones de tiempo.
4. **GPU requerida:** no es viable en entornos sin GPU dedicada, limitando aplicabilidad practica.
5. **No-determinismo GPU:** resultados pueden variar ligeramente entre ejecuciones pese a semilla fija.

---

## 8. FIGURAS DISPONIBLES

| Figura | Contenido | Nota |
|---|---|---|
| figures/transformer_curves.png | Curvas loss/accuracy de v1 | NOTA: son de v1, no de v3 |
| figures/transformer_confusion.png | Confusion matrix normalizada de v1 | NOTA: son de v1 |
| figures/transformer_confusion_absolutos.png | Confusion matrix absolutos de v1 | NOTA: son de v1 |

**Importante:** Las figuras existentes son de v1 (baseline). Para el informe final se necesitan las curvas y confusion de v3 (definitivo). Los datos para generarlas estan en la seccion 4.4 de este documento.

---

## 9. DATOS PARA RECONSTRUIR FIGURAS DE v3

### 9.1 Curvas loss (para graficar)

```
Epoca:  1      2      3      4      5      6      7      8      9      10     11
Train:  1.3339 0.9136 0.8231 0.7808 0.7305 0.6908 0.6468 0.6098 0.5647 0.5365 0.5112
Val:    0.9942 0.8671 0.8314 0.8106 0.7939 0.7804 0.7753 0.7611 0.7661 0.7682 0.7759
                                                          ↑ MEJOR (epoca 8)
```

### 9.2 Curvas accuracy (para graficar)

```
Epoca:  1      2      3      4      5      6      7      8      9      10     11
Train:  0.4891 0.6481 0.6808 0.7003 0.7153 0.7257 0.7535 0.7671 0.7838 0.7968 0.8058
Val:    0.6258 0.6834 0.6922 0.7047 0.7054 0.7146 0.7142 0.7267 0.7388 0.7392 0.7454
```

---

## 10. BIBLIOGRAFIA SUGERIDA

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.
2. Canete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Perez, J. (2020). Spanish Pre-Trained BERT Model and Evaluation Data. PML4DC at ICLR 2020.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
4. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL 2018.
5. Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to Fine-Tune BERT for Text Classification. CCL 2019.
