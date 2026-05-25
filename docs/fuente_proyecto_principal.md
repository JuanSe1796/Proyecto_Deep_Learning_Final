# FUENTE DE VERDAD — Proyecto Principal (PP)

> Este documento contiene TODA la informacion tecnica del Proyecto Principal.
> Sirve como referencia unica para redactar: Articulo IEEE, Reporte tecnico y Presentacion.
> No es necesario consultar notebooks ni JSONs — todo esta aqui.

---

## 1. INFORMACION GENERAL

**Asignatura:** Aprendizaje Profundo — Maestria en IA
**Universidad:** Pontificia Universidad Javeriana, Bogota, 2026
**Profesor:** Ing. Julio Omar Palacio Nino, M.Sc.
**Entrega:** Lunes 25 de mayo de 2026

**Equipo:**
- Felipe Reyes — Modelo Clasico 1 (LSTM) + Articulo IEEE
- Yibby Gonzalez — Preprocesamiento + Modelo Clasico 2 (BiLSTM) + Reporte tecnico
- Daniel Ruiz — Esqueleto entrenamiento + Modelo Combinado (BiLSTM+MultiHead) + Presentacion

**Tarea:** Clasificacion de sentimiento en resenas de hoteles en espanol (5 clases: 1-5 estrellas).

**Entregables PP:**
1. Articulo IEEE (4-6 pag, formato IEEE Conference, PDF)
2. Reporte tecnico detallado (PDF, sin limite de paginas)
3. Notebooks ejecutados (01 a 05)
4. Presentacion (diapositivas)

---

## 2. DATASET: ANDALUSIAN HOTELS REVIEWS

**URL:** https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced
**Idioma:** Espanol (resenas de hoteles de Andalucia, sur de Espana)
**Total de resenas:** 18,172
**Columnas usadas:** `review_text` (texto) y `label` (0-4)

### 2.1 Splits (semilla 42, estratificados)

| Split | Muestras | Porcentaje |
|---|---|---|
| Train | 12,720 | 70% |
| Validacion | 2,726 | 15% |
| Test | 2,726 | 15% |

### 2.2 Distribucion de clases

| Clase | Rating | Muestras (train) | % | Class Weight (balanced) |
|---|---|---|---|---|
| 0 | 1 estrella | 1,174 | 9.2% | 2.1670 |
| 1 | 2 estrellas | 696 | 5.5% | 3.6552 |
| 2 | 3 estrellas | 1,592 | 12.5% | 1.5980 |
| 3 | 4 estrellas | 2,955 | 23.2% | 0.8609 |
| 4 | 5 estrellas | 6,303 | 49.6% | 0.4036 |

- Ratio de desbalance: 9.1:1 (clase 5★ vs clase 2★)
- Desbalance SEVERO: un clasificador trivial que prediga siempre 5★ obtiene 49.6% accuracy pero ~0 F1 macro

### 2.3 Estadisticas de longitud (tokens)

| Estadistica | Valor |
|---|---|
| Media | 78.5 |
| Mediana | 61 |
| Desviacion estandar | 54.9 |
| Minimo | 7 |
| Maximo | 1,416 |
| Percentil 25 | 42 |
| Percentil 75 | 102 |
| Percentil 95 | 148 |
| % resenas > 150 tokens | 4.5% |
| % resenas > 200 tokens | 2.3% |

### 2.4 Longitud por clase

| Clase | Mediana | Media | P95 |
|---|---|---|---|
| 1 estrella | 74 | 92.2 | 206 |
| 2 estrellas | 60 | 79.1 | 158 |
| 3 estrellas | 62 | 81.7 | 156 |
| 4 estrellas | 60 | 77.2 | 142 |
| 5 estrellas | 58 | 73.1 | 132 |

**Hallazgo:** Las resenas negativas (1-2★) tienden a ser mas largas — los clientes detallan sus quejas.

### 2.5 Vocabulario

| Metrica | Valor |
|---|---|
| Vocabulario bruto | 57,231 tipos unicos |
| Tras lowercase | 52,449 tipos |
| Tras clean_text() | 27,967 tipos (reduccion 51.1%) |
| Total tokens (limpio) | 990,353 |

**Cobertura por top-N:**
- Top-1,000: 82.1%
- Top-5,000: 94.1%
- Top-10,000: 97.1%
- Top-20,000: 99.2% **(VOCAB_SIZE elegido)**
- Top-30,000: 100%

### 2.6 Top 20 palabras frecuentes

de (41,672), y (39,692), la (36,802), el (28,742), que (26,337), en (24,493), muy (18,118), a (17,239), un (14,942), con (13,205), es (13,177), **no (12,725)**, una (10,902), para (10,674), las (10,290), hotel (9,876), por (9,146), lo (8,526), del (8,173), los (7,422)

### 2.7 Presencia de "no" por clase

| Clase | % resenas con "no" |
|---|---|
| 1 estrella | 87.0% |
| 2 estrellas | 82.9% |
| 3 estrellas | 70.2% |
| 4 estrellas | 47.8% |
| 5 estrellas | 33.4% |

**Hallazgo critico:** La negacion es un indicador fuerte de sentimiento negativo. Justifica conservar stop words.

### 2.8 Palabras con sesgo negativo extremo (rating_mean <= 1.5)

106 palabras identificadas: fatal (0.753), decepcion (0.750), sucia (0.726), desastre (0.694), sucio (0.659), horrible (0.619), cucarachas (0.553), asco (0.353). Son descriptores de sentimiento legitimos, no sesgo espureo.

### 2.9 Patrones de word clouds
- 1★: palabras negativas (horrible, decepcion, sucia, mal)
- 2★: quejas (caro, pequeno, problemas)
- 3★: vocabulario mixto
- 4★: positivo (bien, bueno, recomendaria)
- 5★: fuertemente positivo (excelente, maravilloso, perfecto)

### 2.10 Limitacion: data leakage

| Tipo | Cantidad |
|---|---|
| Resenas compartidas Train/Val | 868 |
| Resenas compartidas Train/Test | 902 |
| Resenas compartidas Val/Test | 249 |
| Duplicados exactos en Train | 2,674 (21%) |

Esto es una limitacion del dataset original. Las metricas reportadas pueden estar ligeramente infladas por este leakage. Se documenta como trabajo futuro la deduplicacion previa.

---

## 3. PREPROCESAMIENTO

### 3.1 Decisiones de diseno

| Decision | Valor | Justificacion |
|---|---|---|
| Tildes | PRESERVADAS | "mas" ≠ "mas" en espanol |
| Stop words | CONSERVADAS | "no", "muy", "poco" criticas para sentimiento |
| Tokenizacion | Nivel de palabra | Apropiado para RNNs clasicas |
| VOCAB_SIZE | 20,002 (20K + PAD + UNK) | 99.2% cobertura |
| MAX_LEN default | 200 tokens | Cubre >97% de resenas |
| Padding | Post-padding con indice 0 | Estandar para RNNs |
| Truncado | Pre-truncation (primeros max_len tokens) | Inicio de resena mas informativo |
| Indices reservados | 0=PAD, 1=UNK | Convencion estandar |

### 3.2 Pipeline (src/preprocessing.py)

**Funcion `clean_text(text)`:**
1. Conversion a lowercase
2. Normalizacion Unicode (NFKD)
3. Remocion de URLs y emails
4. Remocion de numeros
5. Remocion de caracteres especiales (preserva letras espanolas con tildes)
6. Normalizacion de espacios multiples

**Funcion `build_tokenizer(corpus, vocab_size=20000)`:**
- Construye vocabulario palabra->indice desde corpus de entrenamiento
- Limita a top-N palabras por frecuencia
- Indices: 0=PAD, 1=UNK, 2..N+1=palabras

**Funcion `encode_sequences(texts, tokenizer, max_len)`:**
- Tokeniza texto limpio
- Mapea a indices (palabras fuera de vocabulario -> UNK)
- Aplica truncado a max_len tokens
- Aplica padding hasta max_len

**Funcion `build_splits(df, target_col, seed=42)`:**
- Splits estratificados 70/15/15 con sklearn train_test_split
- Semilla fija = 42
- Guarda archivos CSV en data/

**Funcion `compute_class_weights(y_train)`:**
- Calcula pesos inversamente proporcionales a frecuencia de clase
- Usa sklearn compute_class_weight("balanced", ...)

---

## 4. INFRAESTRUCTURA DE ENTRENAMIENTO

### 4.1 Training (src/training.py)

**Caracteristicas:**
- Optimizador: Adam con weight_decay opcional
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- Early stopping: monitorea val_loss con paciencia configurable
- Gradient clipping: max_norm=1.0
- Device management: CPU/GPU automatico
- Semilla fija: 42 en random, numpy, torch, CUDA

**EarlyStopping:**
- Monitorea val_loss
- Paciencia configurable (epochs sin mejora antes de parar)
- Guarda checkpoint del mejor modelo automaticamente
- Registra mejor epoca

### 4.2 Metricas (src/metrics.py)

**Metricas calculadas:**
- Accuracy
- Precision macro y por clase
- Recall macro y por clase
- F1 macro y por clase
- Matriz de confusion

**Visualizaciones:**
- Curvas de entrenamiento: 2 subplots (loss y accuracy, train vs val)
- Matriz de confusion: heatmap con normalizacion opcional
- Marcador de mejor epoca en las curvas

---

## 5. MODELO CLASICO 1: LSTM (Felipe)

### 5.1 Arquitectura

```
Embedding(20002, 128, padding_idx=0)
  -> LSTM(input=128, hidden=128, 1 capa, unidireccional)
  -> Ultimo hidden state h_n
  -> Dense(128 -> 64, ReLU)
  -> Dropout(0.3)
  -> Dense(64 -> 5)
```

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128,
                 num_classes=5, dropout=0.3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                           num_layers=1, batch_first=True, dropout=0)
        self.fc1     = nn.Linear(hidden_size, 64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        hidden = h_n.squeeze(0)
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
```

### 5.2 Hiperparametros definitivos (v3)

| Parametro | Valor |
|---|---|
| embedding_dim | 128 |
| hidden_size | 128 |
| num_layers | 1 |
| dropout | 0.3 |
| learning_rate | 3e-4 |
| optimizer | Adam |
| batch_size | 64 |
| max_epochs | 30 |
| patience (early stopping) | 5 |
| weight_decay | 0.0 |
| class_weights | NO |
| lr_scheduler | ReduceLROnPlateau |
| gradient_clipping | 1.0 |
| MAX_LEN | 150 |
| VOCAB_SIZE | 20,002 |

### 5.3 Versiones exploradas

| Version | Diferencia | F1 macro |
|---|---|---|
| v1 | Sin pipeline Yibby | descartada |
| v2 | Con pipeline Yibby + class weights | 0.3830 |
| **v3** | **Con pipeline Yibby, SIN class weights** | **0.3996** |
| v4 | sqrt class weights | 0.3919 |

**Conclusion:** Class weights perjudican al LSTM sencillo.

### 5.4 Metricas definitivas

| Metrica | Valor |
|---|---|
| **F1 macro** | **0.3996** |
| **Accuracy** | **0.6042** |
| Precision macro | 0.5639 |
| Recall macro | 0.4408 |
| Parametros | 2,700,933 |
| Tiempo entrenamiento | 321.2s (CPU) |
| Mejor epoca | 19 / 24 |

### 5.5 F1 por clase

| Clase | F1 |
|---|---|
| 1 estrella | 0.686 |
| 2 estrellas | 0.168 |
| 3 estrellas | 0.359 |
| 4 estrellas | 0.022 |
| 5 estrellas | 0.764 |

### 5.6 Matriz de confusion

```
Pred:       1★    2★    3★    4★    5★
Real 1★  [ 203,    9,   33,    0,    7]
Real 2★  [  79,   16,   27,    0,   27]
Real 3★  [  46,   13,  104,    1,  177]
Real 4★  [   6,    3,   49,    7,  568]
Real 5★  [   6,    1,   26,    1, 1317]
```

### 5.7 Historiales de entrenamiento

**Loss:**
- Train: [1.3815, 1.3154, 1.3160, 1.2807, 1.2273, 1.1700, 1.1362, 1.1242, 1.0938, 1.0937, 1.0672, 1.0503, 1.0367, 1.0246, 1.0127, 0.9925, 0.9733, 0.9600, 0.9441, 0.9257, 0.9033, 0.8822, 0.8540, 0.8336]
- Val: [1.2863, 1.3916, 1.2889, 1.2315, 1.1898, 1.1344, 1.0959, 1.1427, 1.1675, 1.0841, 1.1149, 1.0702, 1.0427, 1.0592, 1.0383, 1.0297, 1.0225, 1.0237, 1.0106, 1.0203, 1.0381, 1.0447, 1.0343, 1.0481]

**Accuracy:**
- Train: [0.4529, 0.4854, 0.4942, 0.5012, 0.5056, 0.5340, 0.5494, 0.5539, 0.5631, 0.5663, 0.5735, 0.5752, 0.5823, 0.5843, 0.5910, 0.5980, 0.6053, 0.6112, 0.6152, 0.6256, 0.6350, 0.6425, 0.6573, 0.6675]
- Val: [0.5062, 0.4952, 0.4956, 0.5202, 0.4945, 0.5495, 0.5554, 0.5554, 0.5514, 0.5734, 0.5624, 0.5708, 0.5745, 0.5778, 0.5818, 0.5851, 0.5899, 0.5917, 0.6020, 0.6009, 0.6060, 0.5990, 0.6141, 0.6123]

### 5.8 Analisis

El LSTM unidireccional de 1 capa es el modelo mas simple del proyecto. Su bajo rendimiento (F1=0.3996) se debe a:
- Una sola direccion: no captura contexto posterior (ej. negaciones que aparecen despues del sujeto)
- Sin class weights: predice fuertemente la clase mayoritaria (5★), resultando en F1=0.022 para 4★
- Solo usa el ultimo hidden state, perdiendo informacion de las primeras palabras en resenas largas

Sirve como linea base para demostrar el valor de las mejoras arquitectonicas posteriores.

---

## 6. MODELO CLASICO 2: BiLSTM (Yibby)

### 6.1 Arquitectura

```
Embedding(20002, 128, padding_idx=0)
  -> BiLSTM(input=128, hidden=128/direccion, 2 capas, bidireccional, dropout=0.4)
  -> Global Max Pooling (sobre dimension temporal)
  -> Dropout(0.4)
  -> Dense(256 -> 128, ReLU)
  -> Dense(128 -> 5)
```

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128,
                 n_layers=2, dropout=0.4, n_classes=5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                              num_layers=n_layers, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        pooled = lstm_out.max(dim=1).values  # Global Max Pooling
        out = self.dropout(pooled)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out)
        return logits
```

### 6.2 Hiperparametros definitivos (v1)

| Parametro | Valor |
|---|---|
| embedding_dim | 128 |
| hidden_size | 128 |
| num_layers | 2 |
| dropout | 0.4 |
| learning_rate | 1e-3 |
| optimizer | Adam |
| batch_size | 64 |
| max_epochs | 20 |
| patience (early stopping) | 4 |
| weight_decay | 1e-5 |
| class_weights | NO |
| lr_scheduler | ReduceLROnPlateau |
| gradient_clipping | 1.0 |
| MAX_LEN | 200 |
| VOCAB_SIZE | 20,002 |

### 6.3 Versiones exploradas

| Version | Diferencia | F1 macro | Accuracy |
|---|---|---|---|
| **v1** | **Sin class weights** | **0.5749** | **0.6871** |
| v2 | Con class weights balanced | 0.5989 | 0.6453 |

**Conclusion:** v1 elegida como definitiva (mejor accuracy). v2 tiene mejor F1 macro (+2.4pp) pero peor accuracy (-4.2pp).

### 6.4 Metricas definitivas

| Metrica | Valor |
|---|---|
| **F1 macro** | **0.5749** |
| **Accuracy** | **0.6871** |
| Precision macro | 0.5876 |
| Recall macro | 0.5795 |
| Parametros | 3,253,253 |
| Tiempo entrenamiento | 27,685.9s (CPU) |
| Mejor epoca | 4 / 8 |

### 6.5 F1 por clase

| Clase | F1 |
|---|---|
| 1 estrella | 0.716 |
| 2 estrellas | 0.202 |
| 3 estrellas | 0.564 |
| 4 estrellas | 0.572 |
| 5 estrellas | 0.821 |

### 6.6 Matriz de confusion

```
Pred:       1★    2★    3★    4★    5★
Real 1★  [ 180,   32,   37,    1,    2]
Real 2★  [  45,   22,   74,    8,    0]
Real 3★  [  17,   11,  210,   95,    8]
Real 4★  [   4,    4,   64,  403,  158]
Real 5★  [   5,    0,   19,  269, 1058]
```

### 6.7 Historiales de entrenamiento

**Loss:**
- Train: [1.1734, 0.8666, 0.7279, 0.6287, 0.5289, 0.4459, 0.3571, 0.2410]
- Val: [0.9485, 0.8188, 0.7569, 0.7489, 0.7683, 0.7725, 0.8610, 0.9825]

**Accuracy:**
- Train: [0.5302, 0.6373, 0.6925, 0.7333, 0.7807, 0.8238, 0.8617, 0.9145]
- Val: [0.6034, 0.6566, 0.6845, 0.6893, 0.6988, 0.7172, 0.7256, 0.7388]

### 6.8 Analisis

Mejoras sobre el LSTM (+17.5 puntos F1 macro):
- **Bidireccionalidad:** captura contexto antes y despues de cada palabra. Crucial para negaciones (ej. "no me gusto nada, pero el desayuno estuvo bien")
- **2 capas:** mayor capacidad de representacion jerarquica
- **Global Max Pooling:** extrae las activaciones mas fuertes de toda la secuencia, no solo el ultimo paso temporal
- **Dropout 0.4:** regularizacion mas agresiva compensa mayor capacidad

La clase 4★ pasa de F1=0.022 (LSTM) a F1=0.572 — el modelo ya no colapsa las clases intermedias.

---

## 7. MODELO COMBINADO: BiLSTM + Multi-Head Attention (Daniel)

### 7.1 Enfoque y justificacion

El modelo combinado cumple el requisito del profesor: *"Modelo nuevo/combinado que combine tecnicas o use preprocesamiento avanzado, no es permitido el uso de redes pre-entrenadas."*

Se eligio **BiLSTM + Multi-Head Self-Attention** porque:
1. Combina representacion secuencial (BiLSTM) con mecanismo de atencion (tecnicas combinadas)
2. Multi-Head permite enfocarse simultaneamente en diferentes aspectos linguisticos
3. Proporciona interpretabilidad (pesos de atencion visualizables)
4. No usa redes pre-entrenadas (embeddings entrenados desde cero)

### 7.2 Arquitectura base comun (3 variantes)

```python
class RNNAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256,
                 num_layers=2, num_classes=5, dropout=0.3,
                 pad_idx=0, rnn_type='lstm', attention_module=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = attention_module
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb     = self.dropout(self.embedding(x))
        H, _    = self.rnn(emb)
        ctx, _  = self.attention(H)
        h       = self.dropout(F.relu(self.fc1(ctx)))
        logits  = self.fc2(h)
        return logits
```

### 7.3 Modulos de atencion

**Bahdanau (aditiva):**
```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, H):
        energy  = torch.tanh(self.W(H))
        scores  = self.v(energy).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)
        return context, weights
```

**Multi-Head:**
```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        embed_dim = 2 * hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, H):
        out, weights = self.attn(H, H, H)  # Self-attention
        context = out.mean(dim=1)           # Average pooling
        return context, weights
```

### 7.4 Tres variantes exploradas

| Variante | RNN | Atencion | F1 macro | Accuracy | Params | Tiempo |
|---|---|---|---|---|---|---|
| 1. BiLSTM+Bahdanau | LSTM | Bahdanau | 0.5740 | 0.6471 | 5,191,685 | 129s GPU |
| 2. GRU+Bahdanau | GRU | Bahdanau | 0.5785 | 0.6416 | 4,599,813 | 119s GPU |
| **3. BiLSTM+MultiHead** | **LSTM** | **Multi-Head (4 heads)** | **0.6193** | **0.6548** | **6,110,981** | **304s GPU** |

### 7.5 Diagrama del modelo definitivo

```
Embedding(20002, 128, pad_idx=0)
  -> Dropout(0.4)
  -> BiLSTM(256/direccion, 2 capas, bidireccional, dropout=0.4)
  -> Multi-Head Self-Attention (4 cabezas, embed_dim=512, dropout=0.1)
  -> Average Pooling (sobre secuencia)
  -> Dropout(0.4)
  -> Dense(512 -> 256, ReLU)
  -> Dense(256 -> 5)
```

### 7.6 Hiperparametros definitivos (BiLSTM+MultiHead)

| Parametro | Valor |
|---|---|
| embedding_dim | 128 |
| hidden_size | 256 |
| num_layers | 2 |
| dropout | 0.4 |
| num_heads | 4 |
| learning_rate | 5e-4 |
| optimizer | Adam |
| batch_size | 64 |
| max_epochs | 30 |
| patience (early stopping) | 5 |
| weight_decay | 1e-5 |
| class_weights | SI — balanced [2.167, 3.655, 1.598, 0.861, 0.404] |
| lr_scheduler | ReduceLROnPlateau |
| gradient_clipping | 1.0 |
| MAX_LEN | 200 |
| VOCAB_SIZE | 20,002 |
| **NO usa redes pre-entrenadas** | Cumple restriccion del profesor |

### 7.7 Metricas definitivas

| Metrica | Valor |
|---|---|
| **F1 macro** | **0.6193** |
| **Accuracy** | **0.6548** |
| Precision macro | 0.6086 |
| Recall macro | 0.6454 |
| Parametros | 6,110,981 |
| Tiempo entrenamiento | 303.8s (GPU) |
| Mejor epoca | 11 / 16 |

### 7.8 F1 por clase

| Clase | F1 |
|---|---|
| 1 estrella | 0.756 |
| 2 estrellas | 0.533 |
| 3 estrellas | 0.549 |
| 4 estrellas | 0.480 |
| 5 estrellas | 0.778 |

### 7.9 Matriz de confusion

```
Pred:       1★    2★    3★    4★    5★
Real 1★  [ 180,   62,    8,    2,    0]
Real 2★  [  22,  101,   25,    0,    1]
Real 3★  [   9,   58,  207,   57,   10]
Real 4★  [   5,    6,  129,  319,  174]
Real 5★  [   8,    3,   44,  318,  978]
```

### 7.10 Historiales de entrenamiento

**Loss:**
- Train: [1.4275, 1.2354, 1.1256, 1.0674, 1.0175, 0.9545, 0.9295, 0.8881, 0.8554, 0.8303, 0.7886, 0.7493, 0.7180, 0.6869, 0.6330, 0.5908]
- Val: [1.2520, 1.1076, 1.0870, 1.0610, 1.0691, 1.0239, 0.9601, 1.0006, 1.0236, 0.9590, 0.9298, 0.9970, 1.1355, 1.1715, 1.0961, 1.1191]

**Accuracy:**
- Train: [0.4162, 0.5034, 0.5443, 0.5680, 0.5880, 0.6012, 0.6129, 0.6289, 0.6436, 0.6494, 0.6536, 0.6733, 0.6873, 0.6828, 0.7133, 0.7136]
- Val: [0.5517, 0.5543, 0.6137, 0.6034, 0.6093, 0.6475, 0.6376, 0.6299, 0.6563, 0.6361, 0.6570, 0.5587, 0.6702, 0.6720, 0.6908, 0.6922]

### 7.11 Analisis

Mejoras sobre el BiLSTM simple (+4.4 puntos F1 macro):
- **Multi-Head Self-Attention (4 cabezas):** cada cabeza puede aprender a enfocarse en un aspecto diferente (negaciones, adjetivos, sustantivos de dominio, modificadores)
- **Class weights balanced:** cruciales para este modelo — fuerzan atencion a clases minoritarias. Clase 2★ sube de 0.202 (BiLSTM sin weights) a 0.533
- **Hidden size 256 (vs 128):** mayor capacidad de representacion compensa el dataset pequeno
- **LR mas bajo (5e-4 vs 1e-3):** arquitectura mas compleja requiere convergencia mas estable

Es el **mejor modelo del Proyecto Principal** y el segundo mejor del proyecto completo (superado solo por BETO que usa pre-entrenamiento).

---

## 8. TABLA COMPARATIVA FINAL (PP)

### 8.1 Metricas principales

| Modelo | F1 macro | Accuracy | Precision | Recall | Params | Tiempo |
|---|---|---|---|---|---|---|
| LSTM (Felipe) | 0.3996 | 0.6042 | 0.5639 | 0.4408 | 2.7M | 321s CPU |
| BiLSTM (Yibby) | 0.5749 | 0.6871 | 0.5876 | 0.5795 | 3.3M | 27,686s CPU |
| **BiLSTM+MultiHead (Daniel)** | **0.6193** | **0.6548** | **0.6086** | **0.6454** | **6.1M** | **304s GPU** |

### 8.2 F1 por clase

| Clase | LSTM | BiLSTM | BiLSTM+MultiHead | Mejor |
|---|---|---|---|---|
| 1 estrella | 0.686 | 0.716 | **0.756** | Combinado |
| 2 estrellas | 0.168 | 0.202 | **0.533** | Combinado |
| 3 estrellas | 0.359 | **0.564** | 0.549 | BiLSTM |
| 4 estrellas | 0.022 | **0.572** | 0.480 | BiLSTM |
| 5 estrellas | 0.764 | **0.821** | 0.778 | BiLSTM |

### 8.3 Progresion de mejoras

| Transicion | Delta F1 macro | Factores clave |
|---|---|---|
| LSTM -> BiLSTM | +17.5 pp | Bidireccionalidad, 2 capas, Global Max Pooling |
| BiLSTM -> BiLSTM+MultiHead | +4.4 pp | Multi-Head Attention, class weights, hidden_size 256 |

---

## 9. CONCLUSIONES CLAVE (para documentos)

1. **Complejidad arquitectonica mejora rendimiento:** cada salto de complejidad aporta mejora medible (LSTM < BiLSTM < BiLSTM+MultiHead).

2. **Desbalance severo domina el problema:** sin manejo explicito (class weights), los modelos colapsan hacia la clase mayoritaria (5★). El LSTM sin weights da F1=0.022 en clase 4★.

3. **Class weights: efecto variable por modelo:**
   - LSTM: NO mejoran (empeoran F1 macro)
   - BiLSTM: mejoran F1 (+2.4pp) pero reducen accuracy (-4.2pp)
   - BiLSTM+MultiHead: cruciales para buen rendimiento en clases minoritarias

4. **Clase 2★ siempre la mas dificil:** menor cantidad de datos (5.5%), ambiguedad semantica (ni claramente positiva ni negativa), vocabulario solapado con clases adyacentes.

5. **Resenas negativas mas largas:** las quejas se detallan; las opiniones positivas son mas concisas. Implicacion: MAX_LEN importa mas para clases negativas.

6. **Stop words criticas:** la negacion "no" aparece en 87% de resenas 1★ vs 33% de resenas 5★. Removerla destruiria senal de sentimiento.

7. **Data leakage:** el dataset original contiene duplicados entre splits. Esto infla ligeramente las metricas. Trabajo futuro: deduplicar antes de generar splits.

---

## 10. FIGURAS DISPONIBLES

### EDA
- `figures/eda_distribucion_clases.png` — distribucion de las 5 clases
- `figures/eda_histograma_longitudes.png` — histograma de tokens por resena
- `figures/eda_boxplot_longitud_clase.png` — boxplot longitud vs clase
- `figures/eda_zipf_cobertura.png` — curva Zipf y cobertura por vocabulario
- `figures/eda_top50_words_global.png` — 50 palabras mas frecuentes
- `figures/eda_top30_por_clase.png` — top 30 palabras por clase (5 subplots)
- `figures/eda_wordclouds_por_clase.png` — nubes de palabras por clase
- `figures/eda_class_distribution_weights.png` — distribucion vs class weights

### LSTM
- `figures/lstm_v3_yibby_curves.png` — curvas train/val loss y accuracy (v3 definitivo)
- `figures/lstm_v3_yibby_confusion_matrix.png` — confusion matrix (v3 definitivo)

### BiLSTM
- `figures/bilstm_v1_curves.png` — curvas (v1 definitivo)
- `figures/bilstm_v1_confusion_matrix.png` — confusion matrix (v1 definitivo)

### Modelo Combinado
- `figures/bilstm_multihead_curves.png` — curvas (definitivo)
- `figures/bilstm_multihead_confusion_matrix.png` — confusion matrix (definitivo)
- `figures/combined_f1_por_clase.png` — comparacion F1 por clase entre 3 variantes

### Variantes exploratorias
- `figures/bilstm_bahdanau_curves.png` / `_confusion_matrix.png`
- `figures/gru_bahdanau_curves.png` / `_confusion_matrix.png`

---

## 11. REFERENCIA: MODELO BETO (contexto para articulo)

Para la tabla comparativa completa del articulo IEEE, se incluye el modelo de la Tarea de Investigacion:

| Modelo | Track | F1 macro | Accuracy | Params ent. | Tiempo |
|---|---|---|---|---|---|
| BETO v3 fine-tuned (Sebastian) | TI | 0.6565 | 0.7252 | 21.9M | 1,780s GPU |

BETO supera al mejor modelo PP por 3.7 puntos de F1 macro, demostrando el valor de embeddings contextuales pre-entrenados. Sin embargo, usa 3.6x mas parametros entrenables y requiere GPU obligatoriamente.
