# Plan de Avance Global — Reunion 25 mayo 2026

**Fecha de revision:** 24 mayo 2026 (noche) — INVENTARIO VALIDADO
**Entrega:** Lunes 25 mayo 2026
**Estado general:** URGENTE — modelos listos, faltan documentos

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
| Transformer (Sebastian) | LISTO - F1=0.5873, metricas+figuras integradas | No |
| Articulo IEEE (Felipe) | BORRADOR ~85% en docs/articulo_ieee_borrador.md | **SI — falta PDF final** |
| Reporte tecnico (Yibby) | **NO INICIADO** | **SI** |
| Presentacion (Daniel) | **NO INICIADO** | **SI** |
| Informe investigacion (Sebastian) | **NO INICIADO** | **SI** |

### CAMBIO IMPORTANTE vs. version anterior de este documento

El notebook 05_combined_daniel.ipynb **YA ESTA COMPLETO Y EJECUTADO** con GPU (CUDA).
Daniel entreno 3 variantes de modelo combinado:

| Variante | F1 macro | Accuracy | Params | Tiempo |
|---|---|---|---|---|
| **BiLSTM+MultiHead (GANADOR)** | **0.6193** | 0.6548 | 6,110,981 | 303.8s |
| GRU+Bahdanau | 0.5785 | 0.6416 | 4,599,813 | 119.4s |
| BiLSTM+Bahdanau | 0.5740 | 0.6471 | 5,191,685 | 129.3s |

El BiLSTM+MultiHead es el **MEJOR modelo de todo el proyecto**, superando incluso a BETO (0.6193 vs 0.5873).
Esto cambia la narrativa del proyecto: el modelo combinado sin pre-entrenamiento supera al transformer.

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

**Pendiente:**
- [ ] **ARTICULO IEEE — completar y pasar a PDF formato IEEE Conference**
  - Falta: Abstract, seccion IV-D (modelo combinado de Daniel — datos YA disponibles)
  - Falta: completar fila de tabla comparativa del combinado (datos YA disponibles)
  - Falta: 1 bullet de conclusion sobre modelo combinado
  - Falta: pasar de .md a plantilla IEEE (Overleaf o Word), generar PDF
  - Nota: el 85% del contenido ya esta redactado

**Metricas LSTM definitivo (v3):**
- F1 macro: 0.3996 | Accuracy: 0.6042
- F1 por clase: 1=0.686, 2=0.168, 3=0.359, 4=0.022, 5=0.764
- Parametros: 2,700,933 | Tiempo: 321.2s CPU | Mejor epoca: 19

---

### Yibby Gonzalez

**Completado:**
- [x] Pipeline de preprocesamiento (src/preprocessing.py) — 5 funciones + build_pipeline
- [x] Splits estratificados (train/val/test.csv)
- [x] EDA features y texto (02_eda_yibby.ipynb) — 7/7 items
- [x] Modelo BiLSTM entrenado (2 versiones, v1 definitiva)
- [x] Metricas JSON, curvas, matriz de confusion exportadas

**Pendiente:**
- [ ] **REPORTE TECNICO DETALLADO** (sin limite de paginas) — responsabilidad de Yibby
  - Portada, Resumen ejecutivo, Objetivo
  - EDA completo (todas las graficas — ya estan en figures/)
  - Metodologia (4 arquitecturas con diagramas)
  - Hiperparametros completos en tabla por modelo (datos en JSONs)
  - Resultados (matrices, metricas por clase, curvas — todo en figures/)
  - Conclusiones (respondiendo: los resultados permiten tomar decisiones?)

**Metricas BiLSTM definitivo (v1, sin class weights):**
- F1 macro: 0.5568 | Accuracy: 0.6915
- F1 por clase: 1=0.752, 2=0.212, 3=0.492, 4=0.487, 5=0.840
- Parametros: 3,253,253 | Tiempo: 381.9s CPU | Mejor epoca: 4

---

### Daniel Ruiz

**Completado:**
- [x] Esqueleto de entrenamiento (src/training.py) — train(), evaluate(), EarlyStopping
- [x] Modulo de metricas (src/metrics.py) — compute_metrics, finalize_and_save, graficas
- [x] **MODELO NUEVO/COMBINADO — COMPLETO Y EJECUTADO** (notebook 05)
  - 3 variantes entrenadas: BiLSTM+Bahdanau, GRU+Bahdanau, BiLSTM+MultiHead
  - Implementacion completa: BahdanauAttention, MultiHeadAttentionBlock, RNNAttentionClassifier
  - Metricas JSON para las 3 variantes exportadas
  - Curvas y matrices de confusion para las 3 variantes exportadas
  - Visualizacion de pesos de atencion (interpretabilidad)
  - Tabla comparativa CSV generada
  - Grafica F1 por clase comparativa (combined_f1_por_clase.png)

**Pendiente:**
- [ ] **PRESENTACION DE SLIDES** — no iniciada
  - 12-15 slides: contexto, EDA, 4 modelos, tabla comparativa, conclusiones
  - Coordinar ensayo con el equipo
- [ ] Copiar checkpoints (.pt) de los 3 modelos al repositorio (generados en GPU)

**Modelo definitivo: BiLSTM+MultiHead (MEJOR del proyecto):**
- F1 macro: 0.6193 | Accuracy: 0.6548
- F1 por clase: 1=0.756, 2=0.533, 3=0.549, 4=0.480, 5=0.778
- Parametros: 6,110,981 | Tiempo: 303.8s GPU | Mejor epoca: 11
- Arquitectura: Embedding(128) -> BiLSTM(256, 2 capas) -> MultiHead Attention(4 heads) -> MLP -> 5 clases
- Class weights: SI (balanced)

---

### Sebastian Ruiz

**Completado:**
- [x] Notebook 06_transformer_sebas.ipynb
- [x] PreEntrenado.ipynb ejecutado en Google Colab con GPU Tesla T4
- [x] BETO cased fine-tuned (5 epocas, class weights, lr=2e-5)
- [x] Metricas JSON integradas al repo (results/transformer_metrics.json)
- [x] Curvas y matriz de confusion integradas (figures/transformer_curves.png, transformer_confusion.png)

**Pendiente:**
- [ ] **INFORME DE INVESTIGACION** (max 5 paginas) — no iniciado
  - Introduccion, Metodologia, Resultados (tabla comparativa vs 2 clasicos), Analisis
  - Tabla obligatoria: LSTM vs BiLSTM vs BETO
  - Curvas loss/accuracy del transformer (YA en figures/)
  - Preguntas a responder: supero a los clasicos? por que? costo computacional?
- [ ] **Tema de splits:** Sebastian genero splits PROPIOS en Colab (no uso data/train.csv)
  - Si no hay tiempo para re-ejecutar, documentar discrepancia como limitacion

**Metricas BETO fine-tuned:**
- F1 macro: 0.5873 | Accuracy: 0.6592
- F1 por clase: 1=0.729, 2=0.386, 3=0.514, 4=0.514, 5=0.794
- Parametros totales: 109,854,725 (entrenables: 7,682,309 = 7%)
- Tiempo: 676s en GPU (Tesla T4) | Mejor epoca: 5

---

## TABLA COMPARATIVA FINAL (todos los modelos)

| # | Modelo | F1 macro | Accuracy | Prec macro | Recall macro | F1 2est | F1 3est | Params | Tiempo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **BiLSTM+MultiHead (Daniel)** | **0.6193** | 0.6548 | 0.6086 | 0.6454 | 0.533 | 0.549 | 6.1M | 303.8s GPU |
| 2 | BETO fine-tuned (Sebastian) | 0.5873 | 0.6592 | 0.5810 | 0.5983 | 0.386 | 0.514 | 7.7M ent. | 676s GPU |
| 3 | GRU+Bahdanau (Daniel) | 0.5785 | 0.6416 | 0.5757 | 0.6024 | 0.436 | 0.457 | 4.6M | 119.4s GPU |
| 4 | BiLSTM+Bahdanau (Daniel) | 0.5740 | 0.6471 | 0.5836 | 0.5842 | 0.433 | 0.479 | 5.2M | 129.3s GPU |
| 5 | BiLSTM v1 (Yibby) | 0.5568 | 0.6915 | 0.6130 | 0.5457 | 0.212 | 0.493 | 3.3M | 381.9s CPU |
| 6 | LSTM v3 (Felipe) | 0.3996 | 0.6042 | 0.5639 | 0.4408 | 0.168 | 0.359 | 2.7M | 321.2s CPU |

**Hallazgos clave:**
- El BiLSTM+MultiHead (sin pre-entrenamiento) SUPERA a BETO por 3.2 puntos de F1 macro
- La atencion multi-cabeza es la diferencia mas grande: F1 en clase 2est pasa de 0.212 (BiLSTM puro) a 0.533
- Las clases 2 y 3 estrellas siguen siendo las mas dificiles para todos los modelos
- BETO tiene mejor accuracy pero peor F1 macro que BiLSTM+MultiHead — indica sesgo hacia clase mayoritaria

---

## PLAN DE ACCION PARA HOY 25 MAYO

### Situacion actual (mejor de lo esperado en modelos, peor en documentos)

**LO QUE YA ESTA LISTO (no hay que tocar):**
1. Todos los notebooks ejecutados con outputs (6/6)
2. Todas las metricas JSON (4 definitivos + variantes)
3. Todas las figuras (EDA, curvas, confusion matrices, F1 por clase)
4. Todo el codigo fuente (src/)
5. Borrador del articulo IEEE (~85%)

**LO QUE FALTA (todo son documentos):**

| Documento | Responsable | Datos disponibles | Esfuerzo estimado |
|---|---|---|---|
| Articulo IEEE -> PDF | Felipe | Borrador al 85%, solo falta completar seccion combinado + Abstract + formatear | 2-3 horas |
| Reporte tecnico | Yibby | Todas las graficas y metricas estan listas | 4-5 horas |
| Presentacion slides | Daniel | Todas las figuras listas, tabla comparativa lista | 2-3 horas |
| Informe investigacion | Sebastian | Metricas y curvas listas | 2-3 horas |

### Distribucion de trabajo sugerida

| Persona | Tarea 1 (prioridad) | Tarea 2 |
|---|---|---|
| **Felipe** | Completar articulo IEEE: agregar seccion modelo combinado (datos del notebook 05), Abstract, y pasar a formato IEEE PDF | Revision cruzada |
| **Yibby** | Escribir reporte tecnico completo (Word o LaTeX). Usar graficas de figures/ y datos de JSONs | Revision cruzada |
| **Daniel** | Crear presentacion de slides (12-15 slides). Copiar checkpoints .pt al repo si es posible | Coordinar ensayo de presentacion |
| **Sebastian** | Escribir informe de investigacion (max 5 pag). Tabla: LSTM vs BiLSTM vs BETO | Revision cruzada |

### Datos que Daniel debe proveer a Felipe para el articulo

Para completar la seccion IV-D del articulo IEEE, Felipe necesita (todo esta en el notebook 05):

**Arquitectura del modelo combinado (BiLSTM+MultiHead):**
```
Embedding(20,002, 128, padding_idx=0)
  -> BiLSTM(256 por direccion, 2 capas, dropout=0.4)
  -> Multi-Head Self-Attention (4 cabezas)
  -> Dropout(0.4)
  -> Dense(512 -> 256, ReLU)
  -> Dense(256 -> 5)
```

**Hiperparametros:**
- Optimizer: Adam, lr=5e-4, weight_decay=1e-5
- Batch size: 64, MAX_LEN: 200
- Con class weights (balanced)
- Early stopping: paciencia 5
- LR scheduler: ReduceLROnPlateau

**Justificacion tecnica:**
- Multi-Head Attention (4 cabezas) permite que cada cabeza se especialice en aspectos distintos del texto
- Es el puente conceptual entre RNNs y Transformers (Vaswani et al., 2017)
- No usa redes pre-entrenadas (cumple restriccion del enunciado)

### Datos para el informe de Sebastian

Para la tabla comparativa obligatoria del informe de investigacion:

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Parametros | Tiempo |
|---|---|---|---|---|---|---|
| LSTM (Clasico 1) | 0.6042 | 0.5639 | 0.4408 | 0.3996 | 2,700,933 | 321.2s CPU |
| BiLSTM (Clasico 2) | 0.6915 | 0.6130 | 0.5457 | 0.5568 | 3,253,253 | 381.9s CPU |
| BETO fine-tuned | 0.6592 | 0.5810 | 0.5983 | 0.5873 | 7.68M ent. | 676s GPU |

BETO supera al mejor clasico (BiLSTM) en F1 macro por 3.05 puntos (0.5873 vs 0.5568).
La mejora mas notable es en clases minoritarias: clase 2est (0.386 vs 0.212), clase 3est (0.514 vs 0.492).

### Preguntas pendientes para el equipo

1. **Daniel:** Puede copiar los checkpoints .pt de los modelos combinados? Si pesan mucho, subir a Drive y poner link.
2. **Sebastian:** El informe de investigacion compara SOLO contra los 2 clasicos (LSTM y BiLSTM), NO contra el modelo combinado. Eso es correcto segun el enunciado.
3. **Limitacion de splits de Sebastian:** Decidir si documentar como limitacion o si puede re-ejecutar rapidamente en Colab con los splits del repo.
4. **Formato de entrega:** Confirmar formatos (PDF para articulo/reporte/informe, PPTX+PDF para presentacion).

---

## CHECKLIST FINAL DEL LUNES 25

- [x] Repositorio Git con todo el codigo, notebooks ejecutados
- [x] README.md actualizado con inventario completo
- [x] Metricas JSON de los 4 modelos definitivos
- [x] Figuras de EDA, curvas y confusion matrices
- [ ] Articulo IEEE en PDF (4-6 paginas) — Felipe
- [ ] Reporte tecnico en PDF — Yibby
- [ ] Presentacion en PDF + .pptx — Daniel
- [ ] Informe de investigacion en PDF (5 pag max) — Sebastian
- [ ] Checkpoints de modelos combinados (.pt) — Daniel
- [ ] Los 4 nombres y roles claros en cada documento
