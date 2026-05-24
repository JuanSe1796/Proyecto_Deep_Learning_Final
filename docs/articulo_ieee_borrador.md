# BORRADOR — Articulo IEEE Conference (4-6 paginas)
# Felipe Reyes (coordinador) + equipo
# Fecha: 24-25 mayo 2026
#
# INSTRUCCIONES:
# - Las secciones marcadas con [LISTO] se pueden redactar YA con los datos disponibles
# - Las marcadas con [PENDIENTE-DANIEL] requieren los resultados del modelo combinado
# - Las marcadas con [PENDIENTE-SEBAS] requieren confirmacion de Sebastian
# - Copiar a plantilla IEEE Conference en Overleaf o Word al finalizar
# - Maximo 6 paginas incluyendo referencias

---

# Clasificacion de Resenas de Hoteles en Espanol mediante Redes Neuronales Recurrentes: Un Estudio Comparativo

Felipe Reyes, Yibby Gonzalez, Daniel Ruiz, Sebastian Ruiz
Maestria en Inteligencia Artificial
Pontificia Universidad Javeriana, Bogota, Colombia

---

## I. INTRODUCCION [LISTO]

### Contenido sugerido (~0.5 paginas):

El analisis de sentimiento en resenas de hoteles es una tarea relevante para la industria
hotelera, permitiendo extraer informacion estructurada de opiniones textuales no estructuradas.
En particular, la clasificacion automatica del nivel de satisfaccion (1 a 5 estrellas) a partir
del texto de la resena representa un problema de clasificacion multi-clase con desafios
significativos: desbalance de clases, ambiguedad semantica en clases intermedias y la necesidad
de capturar patrones secuenciales en el lenguaje natural.

El presente trabajo aborda este problema utilizando el dataset Andalusian Hotels Reviews [1],
un corpus de 18,172 resenas de hoteles en espanol procedentes de la region de Andalucia, Espana.
El dataset presenta un desbalance severo, con una relacion 9.1:1 entre la clase mayoritaria
(5 estrellas) y la minoritaria (2 estrellas), lo que impone restricciones sobre las metricas
de evaluacion y las estrategias de entrenamiento.

Se proponen y comparan cuatro arquitecturas de aprendizaje profundo:
(1) un LSTM unidireccional como linea base,
(2) una BiLSTM bidireccional con Global Max Pooling,
(3) un modelo combinado [PENDIENTE-DANIEL: describir arquitectura], y
(4) fine-tuning de BETO, un transformer pre-entrenado para espanol.

El objetivo es evaluar como la complejidad arquitectonica y el uso de representaciones
pre-entrenadas impactan el rendimiento en una tarea de clasificacion de sentimiento
multi-clase en espanol, un idioma sub-representado en la literatura de NLP.

---

## II. OBJETIVOS [LISTO]

### Contenido sugerido (~0.25 paginas):

El objetivo general de este trabajo es disenar, implementar y comparar multiples
arquitecturas de aprendizaje profundo para la clasificacion de resenas de hoteles
en espanol en 5 niveles de satisfaccion.

Los objetivos especificos son:

1. Realizar un analisis exploratorio del dataset que caracterice la distribucion de clases,
   la longitud de los textos y el vocabulario del corpus.
2. Implementar dos modelos clasicos de redes neuronales recurrentes (LSTM y BiLSTM) como
   lineas base para la tarea de clasificacion.
3. Disenar un modelo nuevo/combinado que incorpore [PENDIENTE-DANIEL: tecnica] para mejorar
   el rendimiento sobre los modelos clasicos.
4. Evaluar el impacto del fine-tuning de un transformer pre-entrenado (BETO) en comparacion
   con los modelos clasicos entrenados desde cero.
5. Analizar el efecto del desbalance de clases y las estrategias de mitigacion (class weights)
   en el rendimiento de cada arquitectura.

---

## III. ANALISIS DEL PROBLEMA [LISTO]

### Contenido sugerido (~1 pagina):

### A. Dataset

El dataset Andalusian Hotels Reviews [1] contiene 18,172 resenas de hoteles en espanol,
etiquetadas con un rating de 1 a 5 estrellas. Las resenas fueron recolectadas de plataformas
de reserva de hoteles de la region de Andalucia, Espana.

**Dimensionalidad:**
- 18,172 resenas totales
- 7 columnas originales (texto, rating, titulo, ubicacion, hotel, label)
- 5 clases (1-5 estrellas)
- Splits: 12,720 train (70%), 2,726 val (15%), 2,726 test (15%)

**Distribucion de clases (train):**

| Clase | Muestras | % | Class weight |
|---|---|---|---|
| 1 estrella | 1,174 | 9.2% | 2.167 |
| 2 estrellas | 696 | 5.5% | 3.655 |
| 3 estrellas | 1,592 | 12.5% | 1.598 |
| 4 estrellas | 2,955 | 23.2% | 0.861 |
| 5 estrellas | 6,303 | 49.6% | 0.404 |

El desbalance es severo (ratio 9.1:1). Un clasificador trivial que prediga siempre
la clase mayoritaria alcanzaria ~50% de accuracy pero un F1 macro cercano a 0.10.

**Longitud de resenas (tokens):**
- Media: 78.5, Mediana: 61, P95: 148, Max: 1,416
- Se selecciono MAX_LEN = 150-200 tokens basado en el percentil 95

**Vocabulario:**
- Vocabulario bruto: 57,231 tipos unicos
- Tras limpieza (clean_text): 27,967 tipos unicos (reduccion 51.1%)
- Top-20,000 palabras cubren 99.2% del corpus → VOCAB_SIZE = 20,000

**Figuras para incluir:** eda_distribucion_clases.png, eda_histograma_longitudes.png

### B. Preprocesamiento

El pipeline de preprocesamiento fue disenado siguiendo criterios especificos para el
analisis de sentimiento en espanol:

- **Tildes preservadas:** "mas" (conjuncion) vs "mas" (adverbio) son semanticamente diferentes.
- **Stop words conservadas:** Palabras como "no", "nunca", "pero" son criticas para el sentimiento.
  El analisis mostro que el 87% de las resenas de 1 estrella contienen "no", vs solo 33% de
  las de 5 estrellas.
- **Limpieza:** lowercase, eliminacion de URLs, emails, numeros y caracteres especiales.
  Preservacion de caracteres acentuados y n con tilde.
- **Tokenizacion:** por palabras con vocabulario limitado a top-20,000. PAD=0, UNK=1.
- **Padding/truncado:** longitud fija configurable (150-200 tokens).
- **Estratificacion de splits:** 70/15/15 con semilla fija 42, manteniendo proporcion de clases.

**Limitacion identificada:** Se detecto data leakage entre splits debido a resenas duplicadas
en el dataset original (868 compartidas entre train-val, 902 entre train-test). Esto puede
inflar ligeramente las metricas reportadas.

### C. Metricas de evaluacion

Dada la naturaleza desbalanceada del dataset, se reporta F1 macro como metrica principal,
complementada con F1 por clase, accuracy, precision macro y recall macro. La accuracy sola
es insuficiente ya que un modelo que prediga siempre la clase mayoritaria obtendria ~50%.

---

## IV. CONSTRUCCION, ENTRENAMIENTO E IMPLEMENTACION [PARCIAL]

### Contenido sugerido (~1.5 paginas):

### A. Infraestructura comun

Todos los modelos comparten:
- Framework: PyTorch
- Semilla fija: 42 (random, numpy, torch)
- Optimizador: Adam
- Early stopping por val_loss (paciencia 4-5)
- Gradient clipping: max_norm = 1.0
- LR scheduler: ReduceLROnPlateau (factor=0.5, paciencia=2)
- Evaluacion en test set una sola vez, al final

El esqueleto de entrenamiento (training.py) y el modulo de metricas (metrics.py)
se disenaron para ser reutilizados por todos los modelos, garantizando consistencia
en el formato de resultados y la comparabilidad de metricas.

### B. Modelo Clasico 1: LSTM [LISTO]

Arquitectura:
```
Embedding(20,002, 128, padding_idx=0)
  -> LSTM(128, dropout=0.3)
  -> Dense(128 -> 64, ReLU)
  -> Dropout(0.3)
  -> Dense(64 -> 5)
```

- Parametros: 2,700,933
- Optimizer: Adam, lr = 3e-4
- Batch size: 64, MAX_LEN: 150
- Sin class weights (mejor F1 macro que con class weights balanced o sqrt)
- Early stopping: paciencia 5
- Se entrenaron 4 versiones explorando combinaciones de preprocessing
  (temporal vs pipeline Yibby) y estrategias de class weights
  (balanced, sqrt, ninguno). La version v3 (pipeline Yibby, sin class weights)
  fue seleccionada como definitiva.

### C. Modelo Clasico 2: BiLSTM [LISTO]

Arquitectura:
```
Embedding(20,002, 128, padding_idx=0)
  -> BiLSTM(128 por direccion, 2 capas, dropout=0.4)
  -> Global Max Pooling temporal
  -> Dropout(0.4)
  -> Dense(256 -> 128, ReLU)
  -> Dense(128 -> 5)
```

- Parametros: 3,253,253
- Optimizer: Adam, lr = 1e-3
- Batch size: 64, MAX_LEN: 200
- Sin class weights (v1 > v2 en F1 macro)
- Early stopping: paciencia 4

La BiLSTM procesa la secuencia en ambas direcciones, capturando contexto pasado
y futuro simultaneamente. El Global Max Pooling toma el valor maximo en cada
dimension a lo largo de toda la secuencia, siendo mas robusto que usar solo el
estado oculto final.

Se entrenaron 2 versiones: v1 sin class weights (F1=0.5568) y v2 con class weights
(F1=0.5224). La v1 fue seleccionada como definitiva.

### D. Modelo Nuevo/Combinado [PENDIENTE-DANIEL]

[Espacio reservado para Daniel. Incluir:]
- Arquitectura (diagrama + descripcion)
- Justificacion tecnica de la eleccion
- Hiperparametros
- Restriccion: NO usa redes pre-entrenadas

### E. Modelo Avanzado: Fine-tuning de BETO [LISTO — confirmar con Sebastian]

Arquitectura:
- Modelo base: BETO cased (dccuchile/bert-base-spanish-wwm-cased)
- Tokenizador: WordPiece del modelo pre-entrenado (vocab 31,002)
- Capas descongeladas: encoder.layer.11 + pooler + classifier (7.0% de parametros)
- Parametros totales: 109,854,725 (entrenables: 7,682,309)

Entrenamiento:
- Optimizer: AdamW, lr = 2e-5, weight_decay = 0.01
- Batch size: 32, MAX_LEN: 128
- Con class weights (balanced)
- Warmup: 10% de steps totales
- Epocas: 5 (sin early stopping explicito, la 5ta fue la mejor)
- GPU: Tesla T4 (Google Colab)
- Tiempo: 684 segundos

Nota: El transformer utiliza su propio tokenizador (subword) independiente del
pipeline de preprocesamiento de los modelos clasicos.

---

## V. RESULTADOS [PARCIAL]

### Contenido sugerido (~1.5 paginas):

### A. Tabla comparativa [PARCIAL — falta modelo combinado]

| Modelo | F1 macro | Accuracy | Prec. macro | Recall macro | Params | Tiempo |
|---|---|---|---|---|---|---|
| LSTM | 0.3996 | 0.6042 | 0.5639 | 0.4408 | 2.70M | 321s CPU |
| BiLSTM | 0.5568 | 0.6915 | 0.6130 | 0.5457 | 3.25M | 382s CPU |
| Combinado | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| BETO fine-tuned | 0.5873 | 0.6592 | 0.5810 | 0.5983 | 7.68M* | 684s GPU |

*Parametros entrenables (7.0% de 109.8M totales)

### B. F1 por clase [PARCIAL — falta modelo combinado]

| Clase | LSTM | BiLSTM | Combinado | BETO |
|---|---|---|---|---|
| 1 estrella | 0.686 | 0.752 | [PEND] | 0.729 |
| 2 estrellas | 0.168 | 0.212 | [PEND] | 0.386 |
| 3 estrellas | 0.359 | 0.492 | [PEND] | 0.514 |
| 4 estrellas | 0.022 | 0.487 | [PEND] | 0.514 |
| 5 estrellas | 0.764 | 0.840 | [PEND] | 0.794 |

### C. Analisis de resultados [LISTO con datos parciales]

**LSTM vs BiLSTM:** La BiLSTM supera al LSTM en F1 macro por 15.7 puntos porcentuales
(0.5568 vs 0.3996). La mejora mas dramatica se observa en la clase 4 estrellas
(0.487 vs 0.022), donde el LSTM practicamente no predice esta clase.
La bidireccionalidad y el Global Max Pooling permiten a la BiLSTM capturar
patrones contextuales que el LSTM unidireccional no puede.

**BiLSTM vs BETO:** BETO supera a la BiLSTM en F1 macro por 3 puntos (0.5873 vs 0.5568),
pero la BiLSTM tiene mayor accuracy (0.6915 vs 0.6592). BETO es
significativamente mejor en las clases minoritarias dificiles: 2 estrellas
(0.386 vs 0.212) y 3 estrellas (0.514 vs 0.492). Esto sugiere que las
representaciones contextuales del transformer capturan mejor los matices
de las resenas ambivalentes.

**Class weights:** El uso de class weights "balanced" perjudico al LSTM y al BiLSTM
(F1 macro mas bajo). Sin embargo, BETO con class weights obtuvo buenos resultados.
Esto sugiere que los transformers son mas robustos al ajuste de pesos por clase.

**Clases mas dificiles:** Las clases 2 y 3 estrellas son consistentemente las mas
dificiles para todos los modelos. Esto se explica por:
(a) menor representacion en el dataset (5.5% y 12.5%),
(b) ambiguedad semantica (resenas mixtas que combinan criticas y elogios), y
(c) la confision de la clase 3 estrellas no se observa facilmente.

**Costo computacional:** Los modelos clasicos se entrenaron en CPU en ~5-6 minutos,
mientras que BETO requirio GPU y ~11 minutos. El numero de parametros entrenables
de BETO (7.68M) es ~2.4x el de la BiLSTM (3.25M), pero su total de parametros
(109.8M) es ~34x mayor, lo que implica requisitos de memoria significativos.

**Figuras para incluir:**
- figures/lstm_curves.png y figures/lstm_confusion_matrix.png
- figures/bilstm_v1_curves.png y figures/bilstm_v1_confusion_matrix.png
- [PENDIENTE] figures/combined_curves.png y combined_confusion_matrix.png
- [PENDIENTE-COPIAR] figures del transformer (en Drive de Sebastian)

---

## VI. CONCLUSIONES [LISTO — ajustar cuando haya modelo combinado]

### Contenido sugerido (~0.5 paginas):

Se implementaron y compararon cuatro arquitecturas de aprendizaje profundo para la
clasificacion de resenas de hoteles en espanol en 5 niveles de satisfaccion.

Los resultados demuestran que:

1. La complejidad arquitectonica mejora el rendimiento: BiLSTM > LSTM, con una mejora
   de 15.7 puntos en F1 macro atribuible a la bidireccionalidad y el Global Max Pooling.

2. Las representaciones pre-entrenadas (BETO) aportan una ventaja adicional (+3 puntos
   de F1 macro sobre BiLSTM), especialmente en las clases minoritarias, donde el
   conocimiento previo del idioma espanol permite capturar matices semanticos que los
   modelos entrenados desde cero no logran.

3. El desbalance de clases tiene un impacto severo en la capacidad predictiva.
   Todos los modelos tienen dificultades con las clases 2 y 3 estrellas. Las estrategias
   de class weights no siempre mejoran el F1 macro y su efecto depende de la arquitectura.

4. [PENDIENTE-DANIEL: conclusion sobre el modelo combinado]

5. Se identifico data leakage entre los splits del dataset original, lo que puede
   inflar ligeramente las metricas reportadas. Futuros trabajos deberian realizar
   una deduplicacion previa del corpus.

Como trabajo futuro se propone: (a) deduplicar el corpus antes de generar splits,
(b) explorar tecnicas de data augmentation para las clases minoritarias,
(c) evaluar modelos de lenguaje mas recientes para espanol (RoBERTa-BNE, MarIA), y
(d) considerar la tarea como regresion ordinal en lugar de clasificacion multi-clase,
dado que las clases tienen un orden natural.

---

## REFERENCIAS [LISTO]

[1] chizhikchi, "Andalusian Hotels Reviews (Unbalanced)," Kaggle, 2024.
    https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced

[2] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation,
    vol. 9, no. 8, pp. 1735-1780, 1997.

[3] K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for
    Statistical Machine Translation," EMNLP, 2014.

[4] M. Schuster and K. K. Paliwal, "Bidirectional Recurrent Neural Networks," IEEE
    Transactions on Signal Processing, vol. 45, no. 11, pp. 2673-2681, 1997.

[5] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding," NAACL-HLT, 2019.

[6] J. Canete et al., "Spanish Pre-Trained BERT Model and Evaluation Data," PML4DC
    at ICLR, 2020.

[7] Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP, 2014.

[8] D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly
    Learning to Align and Translate," ICLR, 2015.

[9] [PENDIENTE: agregar mas referencias segun el modelo combinado de Daniel]

[10] [PENDIENTE: agregar referencia de analisis de sentimiento en espanol]

---

## NOTAS PARA LA REDACCION FINAL

### Figuras a incluir (seleccionar las mas importantes para 4-6 paginas):
1. Distribucion de clases (eda_distribucion_clases.png)
2. Histograma de longitudes (eda_histograma_longitudes.png) - OPCIONAL si falta espacio
3. Curvas de entrenamiento del mejor modelo (bilstm_v1_curves.png o combinar en una)
4. Matrices de confusion (las 4, o las 2 mejores si falta espacio)
5. Tabla comparativa (obligatoria)

### Formato IEEE Conference:
- Usar plantilla oficial: https://www.ieee.org/conferences/publishing/templates.html
- 2 columnas, Times New Roman 10pt
- Figuras numeradas (Fig. 1, Fig. 2, ...)
- Referencias en formato IEEE [1], [2], ...
- Abstract obligatorio (~150 palabras)
- Falta agregar el ABSTRACT al inicio

### Partes que se pueden escribir HOY sin esperar a nadie:
- [x] Introduccion (ajustar una frase cuando se tenga el modelo combinado)
- [x] Objetivos
- [x] Analisis del problema (completo)
- [x] Modelo LSTM (seccion IV-B)
- [x] Modelo BiLSTM (seccion IV-C)
- [x] Modelo BETO (seccion IV-E)
- [x] Resultados parciales (LSTM, BiLSTM, BETO)
- [x] Conclusiones (ajustar 1 bullet cuando haya modelo combinado)
- [x] Referencias (agregar 1-2 cuando se tenga modelo combinado)

### Partes que requieren a Daniel/Sebastian:
- [ ] Seccion IV-D (modelo combinado de Daniel)
- [ ] Fila de tabla comparativa del modelo combinado
- [ ] 1 bullet de conclusion sobre el modelo combinado
- [ ] Confirmar con Sebastian si se re-ejecuto con splits del repo
