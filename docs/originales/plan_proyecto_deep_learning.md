# Plan de Trabajo — Proyecto Final Deep Learning (RNN)

> **Equipo de 4 personas** · Felipe Reyes, Yibby González, Daniel Ruiz, Sebastián Ruiz
> **Entrega y presentación:** lunes 25 de mayo de 2026
> **Asignatura:** Aprendizaje Profundo · **Profesor:** Ing. Julio Omar Palacio Niño, M.Sc.

---

## 0. Cómo usar este documento

Este documento es la **fuente única de verdad** del plan de trabajo. Está pensado para que cada persona pueda:

1. **Leer su sección** y entender exactamente qué tiene que entregar y cuándo.
2. **Pegarlo (completo o por secciones) en su LLM de preferencia** (ChatGPT, Claude, Gemini, etc.) para hacer preguntas técnicas concretas. Cada sección de persona es autocontenida e incluye todo el contexto necesario para que el LLM ayude sin tener que leer los PDFs originales.
3. **Validar el progreso** contra los checklists de cada hito.

Si tu LLM te pregunta "¿qué dataset es?" o "¿cuál es la arquitectura del modelo X?", la respuesta está en este documento. Si algo no está claro, mejóralo aquí en lugar de que cada quien improvise.

> **Tip de uso con LLM:** para preguntas técnicas, copia tu sección + la sección de "Contexto del proyecto" + la sección de "Convenciones del equipo". Eso es suficiente para que cualquier LLM moderno te ayude bien.

---

## 1. Contexto del proyecto

Tenemos **dos entregas separadas pero conectadas**, definidas en dos PDFs distintos del profesor:

### 1.1 Proyecto Principal (Felipe, Yibby, Daniel)

> Cita del enunciado (`Proyecto_2026-2.pdf`):
>
> *"El proyecto final de la asignatura es la oportunidad para explorar un problema de clasificación de Deep Learning en un conjunto de datos del mundo real. El objetivo es entregar una solución completa y funcional que demuestre la profundidad técnica y el alcance de la exploración realizada."*

**Lo que hay que construir:**
- 2 modelos **clásicos** de RNN.
- 1 modelo **nuevo/combinado** que use técnicas combinadas o preprocesamiento avanzado. **No está permitido usar redes pre-entrenadas** en este modelo.
- EDA completo del dataset.
- Artículo en formato IEEE Conference (4–6 páginas).
- Reporte técnico detallado (sin límite de páginas).
- Notebook con todo el código.
- Presentación.

> Cita del enunciado (sección 3):
>
> *"En esta sección se deben proponer 2 modelos clásicos (CNN, o RNN según el problema). Modelo nuevo/combinado (que combine técnicas o use preprocesamiento avanzado, no es permitido el uso de redes pre-entrenadas)."*

### 1.2 Tarea de Investigación (Sebastián)

> Cita del enunciado (`Proyecto_2026-3.pdf`):
>
> *"Esta fase es una tarea de investigación técnica que complementa el proyecto principal. Corresponde a la implementación del 'nuevo modelo' solicitado en el proyecto, el cual debe demostrar una mayor profundidad técnica. El objetivo es realizar un análisis experimental comparativo que enfrente los modelos clásicos contra arquitecturas más avanzadas o técnicas modernas de Deep Learning."*

**Lo que hay que construir:**
- 1 modelo avanzado: **fine-tuning de un Transformer pre-entrenado** (BETO, BERTIN o RoBERTa en español).
- Comparación experimental contra el **mejor de los 2 modelos clásicos** del proyecto principal (no contra el modelo nuevo/combinado).
- Notebook con la implementación.
- Informe de investigación separado (máx. 5 páginas).

> Cita del enunciado (sección "Tarea Específica para el Enfoque 2", apartado B):
>
> *"Implementar un modelo usando Fine-Tuning de un Transformer. Detalle: Deberán usar un modelo de lenguaje pre-entrenado (ej. BETO, RoBERTa) y realizar fine-tuning de (al menos) la última capa del transformer para la tarea de clasificación."*

### 1.3 Por qué son entregas separadas

Aunque están conectadas (la Tarea de Investigación reutiliza las métricas de los modelos clásicos), son **dos PDFs distintos del profesor con dos sets de entregables distintos**. Esto significa:

- El **modelo nuevo/combinado** del proyecto principal (Daniel) y el **modelo avanzado del transformer** (Sebastián) son **dos modelos diferentes**, con diferentes restricciones (uno NO puede usar pre-entrenado, el otro SÍ debe usarlo).
- El **informe de investigación** de la Tarea de Investigación es un PDF separado del artículo IEEE y del reporte técnico del proyecto principal.
- La Tarea de Investigación compara solo contra **los 2 clásicos**, no contra el modelo nuevo/combinado.

---

## 2. El dataset: Andalusian Hotels Reviews

**URL:** https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced

**Características relevantes (a confirmar en EDA):**
- **Idioma:** español.
- **Dominio:** reseñas de hoteles de Andalucía (sur de España).
- **Tarea:** clasificación supervisada (probablemente por puntuación/rating o por sentimiento).
- **Característica clave:** el dataset está **desbalanceado** (lo dice el nombre: `unbalanced`). Esto debe influir en:
  - La elección de métricas (no solo accuracy: F1 macro y por clase).
  - Estrategias de manejo de desbalance: class weights, oversampling (SMOTE), focal loss.
  - Estratificación de los splits.

**Lo que toca confirmar en el EDA (Felipe + Yibby):**
- Número exacto de reseñas y de clases.
- Distribución de clases.
- Distribución de longitudes (en tokens y caracteres).
- Tamaño de vocabulario.
- Posibles sesgos (¿hay nombres de hoteles que filtran trivialmente la clase?).

---

## 3. El equipo y los roles

| Persona | Modelo a su cargo | Documento que coordina | Track |
|---|---|---|---|
| **Felipe Reyes** | Modelo Clásico 1 (LSTM o GRU) | Artículo IEEE | Proyecto Principal |
| **Yibby González** | Modelo Clásico 2 (BiLSTM) + Pipeline de preprocesamiento | Reporte técnico | Proyecto Principal |
| **Daniel Ruiz** | Modelo Nuevo/Combinado + Esqueleto de entrenamiento común | Presentación | Proyecto Principal |
| **Sebastián Ruiz** | Modelo Avanzado (Fine-tuning de Transformer) | Informe de investigación | Tarea de Investigación |

**Principio de trabajo:** cada persona es **dueña** de un modelo y de un documento, pero todos contribuyen con párrafos sobre su propio modelo a los demás documentos cuando aplique. Los "dueños" consolidan, dan formato y revisan coherencia.

---

## 4. Convenciones del equipo (acordadas el día 1)

### 4.1 Estructura del repositorio

```
deep-learning-rnn/
  data/                          # Dataset crudo + splits (en .gitignore si pesa mucho)
    train.csv                    # Yibby genera estos 3 archivos con semilla 42
    val.csv
    test.csv
  notebooks/
    01_eda_felipe.ipynb          # EDA dimensionalidad + clases
    02_eda_yibby.ipynb           # EDA features + texto
    03_lstm_felipe.ipynb         # Modelo Clásico 1
    04_bilstm_yibby.ipynb        # Modelo Clásico 2
    05_combined_daniel.ipynb     # Modelo Nuevo/Combinado
    06_transformer_sebas.ipynb   # Modelo Avanzado (Tarea de Investigación)
  src/
    preprocessing.py             # Yibby - pipeline de texto reusable
    training.py                  # Daniel - esqueleto train() reusable
    metrics.py                   # Daniel - generador de JSON estandarizado
  results/                       # JSON de métricas + checkpoints de modelos
    lstm_metrics.json
    bilstm_metrics.json
    combined_metrics.json
    transformer_metrics.json
  figures/                       # Gráficas exportadas (PNG) para los documentos
  docs/                          # Borradores de los entregables escritos
  README.md                      # Instrucciones para reproducir todo
```

### 4.2 Formato JSON estandarizado de métricas

Todos los modelos guardan sus resultados en este formato. Así, armar la tabla comparativa final es leer 4 archivos y concatenar.

```json
{
  "model_name": "bilstm_v1",
  "owner": "Yibby",
  "track": "PP",
  "config": {
    "embedding_dim": 128,
    "hidden_size": 128,
    "dropout": 0.3,
    "use_class_weights": true,
    "n_params": 542000
  },
  "metrics": {
    "accuracy": 0.85,
    "precision_macro": 0.82,
    "recall_macro": 0.81,
    "f1_macro": 0.81,
    "f1_per_class": {"1": 0.70, "2": 0.65, "3": 0.78, "4": 0.84, "5": 0.90},
    "confusion_matrix": [[10, 2, 0], [1, 15, 3], [0, 2, 20]]
  },
  "training": {
    "epochs_run": 12,
    "best_epoch": 8,
    "training_time_seconds": 340,
    "loss_history": [],
    "val_loss_history": [],
    "acc_history": [],
    "val_acc_history": []
  }
}
```

### 4.3 Reglas técnicas

- **Semilla aleatoria fija = 42** en `numpy`, `torch`/`tensorflow` y `random`. Sin excepciones.
- **Splits idénticos para todos:** Yibby los genera y los guarda como archivos. Los demás los **cargan**, no los re-generan.
- **Estratificación obligatoria** en los splits (mantener proporción de clases).
- **Métricas reportadas:** siempre F1 macro y por clase (no solo accuracy: el dataset está desbalanceado).
- **Cada notebook empieza** con celda de configuración: imports, semilla, paths comunes.
- **Antes de hacer push:** ejecutar el notebook completo y dejar las salidas guardadas (para que el resto vea resultados sin tener que correr todo).

### 4.4 Comunicación

- Grupo de WhatsApp/Slack con check-in diario corto (3 líneas: qué hice ayer, qué hago hoy, en qué estoy bloqueado).
- Reuniones obligatorias en cada hito (8 mayo, 14 mayo, 21 mayo).
- Decisiones técnicas grandes (qué arquitectura, qué embeddings, qué transformer base) se documentan en el `README.md`.

---

## 5. Cronograma general

| Fechas | Actividades | Responsables |
|---|---|---|
| Lun 4 – Mar 5 | Bloque 0: repositorio, dataset, plantilla Colab, semilla, formato de métricas. Reunión kickoff martes. | Todos |
| Mié 6 – Vie 8 | EDA. Pipeline preprocesamiento. Esqueleto entrenamiento. Smoke test transformer. | Todos en sus tracks |
| **Vie 8 noche · ★ HITO 1** | **Cimientos listos:** splits, pipeline, esqueleto, transformer corriendo en GPU. | Validación grupal |
| Sáb 9 – Dom 10 | Primer entrenamiento end-to-end de los 4 modelos. | Felipe, Yibby, Daniel, Sebastián |
| Lun 11 – Mié 13 | Iteración: hiperparámetros, regularización, manejo desbalance, early stopping. | Todos |
| **Jue 14 noche · ★ HITO 2** | **Modelos congelados** con métricas finales en JSON. Reunión revisión 30 min. | Todos |
| Vie 15 – Dom 17 | Tabla comparativa final. Identificar mejor clásico. Arranque redacción en paralelo. | Felipe (artículo), Yibby (reporte), Daniel (slides), Sebastián (informe) |
| Lun 18 – Mié 20 | Secciones de resultados y conclusiones en cada documento. | Todos |
| **Jue 21 noche · ★ HITO 3** | **Borradores 1.0 completos** de los 4 entregables. | Todos |
| Vie 22 – Sáb 23 | Revisión cruzada. Ensayo presentación con cronómetro. PDFs finales sábado noche. | Todos |
| Dom 24 | Día buffer. Ensayo final. | Todos |
| **Lun 25** | **Entrega y presentación.** | Todos |

---

# Parte II — Plan detallado por persona

> **Cada persona puede pegar su sección + las secciones 1, 2, 4 y 6 (apéndice técnico) en su LLM y avanzar.**

---

## 6. Felipe Reyes — EDA + Modelo Clásico 1 + Artículo IEEE

**Track:** Proyecto Principal
**Modelo a tu cargo:** Modelo Clásico 1 (LSTM o GRU sencillo)
**Documento que coordinas:** Artículo IEEE (4–6 páginas, formato IEEE Conference)

### 6.1 Tu rol en el equipo

Eres responsable de **dos cosas grandes**:

1. **El EDA de dimensionalidad y de la variable objetivo** (con Yibby cubriendo features y texto).
2. **El modelo más sencillo de RNN**, que sirve como línea base contra la cual se comparan los otros tres modelos.
3. **El artículo IEEE**, que es el entregable más visible del proyecto principal.

### 6.2 Setup (Lun 4 – Mar 5)

- [ ] Validar que tu entorno Colab/Jupyter funciona con GPU (verifica con `torch.cuda.is_available()` o `tf.config.list_physical_devices('GPU')`).
- [ ] Descargar el dataset y hacer una primera inspección visual (5 minutos: ¿cuántas filas tiene?, ¿qué columnas?, ¿qué pinta tiene una reseña?).
- [ ] Coordinar con Yibby el reparto del EDA: ella se queda con frecuencias de palabras, nubes, sesgos; tú te quedas con dimensionalidad y variable objetivo.

### 6.3 EDA (Mié 6 – Vie 8)

Lo que el enunciado pide explícitamente para esta sección:

> Cita de `Proyecto_2026-2.pdf`, sección 2:
>
> *"Análisis de Dimensionalidad: Descripción del tamaño del dataset (número de muestras, número de características). Para texto (RNN): Longitud de las reseñas (media, máx, mín), tamaño del vocabulario."*
>
> *"Análisis de la Variable Objetivo (Clases): Identificación clara de la variable a predecir. Análisis de balance de clases (distribución gráfica)."*

**Tareas concretas:**

- [ ] Calcular número total de reseñas, número de clases, número de columnas/features.
- [ ] **Longitud de reseñas** (en tokens y en caracteres):
  - Estadísticas: media, mediana, mínimo, máximo, desviación, percentil 25, 75, 95.
  - Histograma de longitudes.
  - Boxplot por clase (¿las reseñas negativas son más largas que las positivas, por ejemplo?).
- [ ] **Tamaño de vocabulario:**
  - Total de palabras únicas antes de limpieza.
  - Tras limpieza básica (lowercase, sin puntuación).
  - Curva de frecuencia (palabras más frecuentes vs. cobertura del corpus, ley de Zipf).
- [ ] **Variable objetivo:**
  - Identificar exactamente qué se va a predecir (nombre de la columna, tipo: si es escala 1–5, si es positivo/negativo, etc.).
  - Distribución de clases en una **gráfica de barras + tabla con conteo y porcentaje**.
  - Documentar el grado de desbalance (ratio entre clase más frecuente y menos frecuente).

**Notebook:** `notebooks/01_eda_felipe.ipynb`

**Al final del viernes 8:** publica un resumen de hallazgos clave (5–7 bullets) en el grupo del equipo. Esto le sirve a Yibby para decidir el preprocesamiento, a Daniel para diseñar el modelo nuevo, y a Sebastián para configurar bien el transformer.

### 6.4 Modelo Clásico 1 (Sáb 9 – Jue 14)

**Arquitectura sugerida:**

```
Embedding(vocab_size, embedding_dim=128)
    → LSTM(hidden_size=128, dropout=0.3)
    → Dense(hidden_size=64, activation='relu')
    → Dropout(0.3)
    → Dense(num_classes, activation='softmax')
```

**Decisiones a tomar y documentar:**
- ¿LSTM o GRU? (cualquiera está bien, pero mantén una y justifica.)
- ¿Embedding entrenable desde cero o pre-cargado con FastText/Word2Vec en español? Coordina con Yibby.
- ¿Cuántas capas? (1 está bien para el modelo "sencillo".)

**Hiperparámetros base sugeridos:**
- `embedding_dim`: 100 o 128
- `hidden_size`: 64 o 128
- `dropout`: 0.3
- `batch_size`: 32 o 64
- `optimizer`: Adam con `lr=1e-3`
- `epochs`: hasta 30 con early stopping (paciencia 5)

**Tareas:**

- [ ] Importar el pipeline de preprocesamiento de Yibby (`from src.preprocessing import build_pipeline`).
- [ ] Cargar los splits desde `data/train.csv`, `val.csv`, `test.csv` (no regenerarlos).
- [ ] Importar el esqueleto de entrenamiento de Daniel (`from src.training import train`).
- [ ] Iterar **2–3 configuraciones** de hiperparámetros (no más, hay tiempo limitado).
- [ ] Entrenar la versión final, evaluar en test set.
- [ ] Generar:
  - JSON de métricas en `results/lstm_metrics.json` (formato de la sección 4.2).
  - Curvas loss/accuracy de train vs. val en `figures/lstm_curves.png`.
  - Matriz de confusión en `figures/lstm_confusion.png`.
  - Checkpoint del mejor modelo en `results/lstm_best.pt`.
- [ ] Reportar tiempo de entrenamiento y número de parámetros del modelo.

### 6.5 Artículo IEEE (Vie 15 – Jue 21)

> Cita de `Proyecto_2026-2.pdf`, sección 4:
>
> *"Formato: Estrictamente en formato IEEE Conference (PDF). Extensión: Mínimo 4 y máximo 6 páginas. Contenido: Introducción, Objetivos, Análisis del problema, Construcción/entrenamiento e implementación, Resultados, Conclusiones."*

**Plantilla:** la oficial de IEEE Conference. Disponible en Overleaf (busca "IEEE Conference Template") o desde la web de IEEE: https://www.ieee.org/conferences/publishing/templates.html

**Estructura sugerida:**

| Sección | Páginas aprox. | Cuándo redactar |
|---|---|---|
| Introducción | 0.5 | Vie 15 (no depende de resultados) |
| Objetivos | 0.25 | Vie 15 |
| Análisis del problema (incluye EDA breve) | 1 | Sáb 16 |
| Construcción, entrenamiento e implementación (las 4 arquitecturas brevemente) | 1.5 | Dom 17 – Lun 18 |
| Resultados (tabla comparativa, curvas, matrices) | 1.5 | Mar 19 – Mié 20 |
| Conclusiones | 0.5 | Jue 21 |
| Referencias | en cualquier momento | Vie 15 – Jue 21 |

**Bibliografía mínima:** 8–12 referencias. Sugerencias:
- Hochreiter & Schmidhuber 1997 (LSTM original).
- Cho et al. 2014 (GRU).
- Devlin et al. 2018 (BERT).
- Cañete et al. 2020 (BETO).
- Papers de análisis de sentimiento en español.
- Página del dataset en Kaggle (cita la URL).

**Tareas:**

- [ ] Crear plantilla en Overleaf con formato IEEE Conference.
- [ ] Redactar las secciones que **no dependen de resultados** primero (Vie 15): Introducción, Objetivos, parte del Análisis del problema.
- [ ] Recopilar bibliografía en un archivo `.bib`.
- [ ] Pedir a Yibby, Daniel y Sebastián los párrafos sobre sus propios modelos para integrarlos en la sección de Construcción.
- [ ] Integrar la tabla comparativa y figuras una vez que Daniel/Sebastián las generen.
- [ ] Verificar al cierre: 4–6 páginas exactas, formato IEEE, figuras numeradas, citas en estilo IEEE.

### 6.6 Lo que necesitas de los demás

- **De Yibby:** los splits guardados como archivo, el módulo de preprocesamiento, párrafo descriptivo de su modelo BiLSTM.
- **De Daniel:** el esqueleto de entrenamiento, párrafo descriptivo de su modelo nuevo/combinado.
- **De Sebastián:** sus métricas (para incluirlas como referencia/contexto en el artículo) y párrafo descriptivo del transformer.

### 6.7 Lo que los demás necesitan de ti

- **Resumen de hallazgos del EDA** al final del viernes 8.
- **Métricas y figuras del modelo LSTM** al final del jueves 14 (van en todos los documentos).
- **Borrador del artículo IEEE** el jueves 21.

### 6.8 Prompts útiles para tu LLM

- *"Aquí está mi sección del plan + el contexto del proyecto. Genera código Python para entrenar una LSTM en PyTorch para clasificación multi-clase con class weights, usando un pipeline de preprocesamiento que ya está empaquetado."*
- *"Revísame este texto de la sección de Introducción del artículo IEEE y dime si cumple con el estilo IEEE y si la primera frase engancha bien."*
- *"Aquí están mis estadísticas de longitud de reseñas: [...]. ¿Qué interpretación puedo dar de esto y cómo lo redacto en una sola frase para el artículo?"*

---

## 7. Yibby González — Preprocesamiento + Modelo Clásico 2 + Reporte técnico

**Track:** Proyecto Principal
**Modelo a tu cargo:** Modelo Clásico 2 (BiLSTM)
**Documento que coordinas:** Reporte técnico detallado (sin límite de páginas)

### 7.1 Tu rol en el equipo

Eres responsable de **tres cosas grandes**:

1. **El pipeline de preprocesamiento**, que es la base sobre la que entrenan Felipe, Daniel y tú misma. Si esto se atrasa, los tres se atrasan.
2. **El EDA de features y texto** (con Felipe cubriendo dimensionalidad y clases).
3. **El modelo BiLSTM**, segunda línea base.
4. **El reporte técnico extendido**, el documento más largo y detallado del proyecto.

### 7.2 Setup (Lun 4 – Mar 5)

- [ ] Crear repositorio Git (GitHub o GitLab) y dar acceso a los 4.
- [ ] Crear la estructura de carpetas de la sección 4.1 con archivos `.gitkeep` en las vacías.
- [ ] Crear `README.md` inicial con la descripción del proyecto, los 4 nombres y los enlaces relevantes.
- [ ] Configurar `.gitignore` para ignorar checkpoints pesados, datos crudos si pesan, `__pycache__`, etc.
- [ ] Coordinar con Felipe el reparto del EDA.

### 7.3 EDA — Features y texto (Mié 6 – Vie 8)

Lo que el enunciado pide para esta sección:

> Cita de `Proyecto_2026-2.pdf`, sección 2:
>
> *"Análisis de Características (Features): Descripción de las características (tipo de dato, significado). Análisis estadístico (media, mediana, desviación) para datos numéricos. Análisis de frecuencia para datos categóricos. Visualizaciones relevantes (histogramas, diagramas de dispersión, nubes de palabras, etc.)."*

**Tareas concretas:**

- [ ] **Frecuencia de palabras:**
  - Top 50 palabras más frecuentes en todo el corpus (excluyendo y con stopwords, comparar).
  - Top 30 palabras más frecuentes **por clase**.
  - Identificar palabras "discriminativas": que aparecen mucho en una clase y poco en otras.
- [ ] **Nubes de palabras** (`wordcloud` en Python) por clase.
- [ ] **Detección de sesgos:**
  - ¿Hay nombres de hoteles o ciudades en las reseñas? ¿Filtran trivialmente la clase?
  - ¿Hay reseñas duplicadas o casi-duplicadas?
- [ ] **Stop words:**
  - Decidir si remover stop words en español (lista de NLTK o spaCy).
  - **Argumento típico para NO removerlas:** las palabras como "no", "muy", "poco" son cruciales para sentimiento.
- [ ] **Caracteres especiales:**
  - ¿Hay emojis? ¿URLs? ¿Menciones?
  - Decidir si removerlos o codificarlos.

**Notebook:** `notebooks/02_eda_yibby.ipynb`

### 7.4 Pipeline de preprocesamiento — TAREA CRÍTICA (Mié 6 – Vie 8 noche)

**Esto es lo más importante de tu primera semana. Sin esto, nadie puede entrenar.**

**Funcionalidades a implementar en `src/preprocessing.py`:**

```python
# Esqueleto sugerido
def clean_text(text: str) -> str:
    """Limpia un texto: lowercase, manejo de tildes/puntuación, etc."""
    pass

def build_tokenizer(corpus: list[str], vocab_size: int = 20000):
    """Construye un tokenizador palabra→índice limitado a top-N."""
    pass

def encode_sequences(texts: list[str], tokenizer, max_len: int) -> np.ndarray:
    """Tokeniza, encodea y aplica padding/truncado."""
    pass

def build_splits(df: pd.DataFrame, target_col: str, seed: int = 42):
    """Splits estratificados 70/15/15. Guarda en data/."""
    pass

def compute_class_weights(y_train) -> dict:
    """Class weights inversamente proporcionales a la frecuencia."""
    pass

def build_pipeline(config: dict):
    """Función orquestadora que cualquiera puede llamar."""
    pass
```

**Decisiones a tomar y documentar:**

- [ ] **Limpieza:** lowercase sí/no (sí), tildes mantener/remover (mantener — son significativas en español), puntuación remover sí/no (probablemente sí).
- [ ] **Tokenización:** ¿por palabras (con vocabulario top-N) o por subwords (BPE)? Empezar por palabras.
- [ ] **Tamaño de vocabulario:** elegir basándote en la curva de frecuencia. Algo entre 10K y 30K suele estar bien.
- [ ] **Longitud máxima** para padding/truncado: usar el **percentil 95** que calculó Felipe (típicamente algo entre 100 y 300 tokens).
- [ ] **Splits:** estratificados 70/15/15 con `train_test_split(stratify=y, random_state=42)` aplicado dos veces.
- [ ] **Embeddings:** ¿entrenables desde cero o pre-cargados? Empezar con entrenables; si hay tiempo, comparar con FastText español pre-entrenado.

**Entregable de esta tarea:**

- [ ] `src/preprocessing.py` empaquetado y probado.
- [ ] `data/train.csv`, `data/val.csv`, `data/test.csv` generados con semilla 42.
- [ ] Un README en `src/` que muestre cómo usar el pipeline en 3 líneas.

### 7.5 Modelo Clásico 2 — BiLSTM (Sáb 9 – Jue 14)

**Arquitectura sugerida:**

```
Embedding(vocab_size, embedding_dim=128)
    → BiLSTM(hidden_size=128, dropout=0.3)
    → Dense(hidden_size=64, activation='relu')
    → Dropout(0.3)
    → Dense(num_classes, activation='softmax')
```

**Diferencia con el modelo de Felipe:**
- Bidireccional (lee la secuencia de izquierda a derecha y de derecha a izquierda). Mayor capacidad expresiva.
- Justificación técnica: en reseñas de hoteles, el contexto antes y después de una palabra ayuda a interpretar correctamente (ej. "no me gustó nada, pero el desayuno estuvo bien" — el "no me gustó" inicial es ambiguo hasta que se ve el contraste).

**Tareas:**

- [ ] Diseñar e implementar la arquitectura.
- [ ] Entrenar **2 versiones**: con y sin class weights. Comparar.
- [ ] Si hay tiempo, probar con embeddings pre-cargados vs. entrenables.
- [ ] Generar:
  - `results/bilstm_metrics.json`
  - `figures/bilstm_curves.png`
  - `figures/bilstm_confusion.png`
  - `results/bilstm_best.pt`

### 7.6 Reporte técnico (Vie 15 – Jue 21)

> Cita de `Proyecto_2026-2.pdf`, sección 5:
>
> *"Este documento es el reporte técnico detallado de todo el proyecto con mayor detalle que en artículo. A diferencia del artículo, este reporte no tiene límite de páginas y debe incluir en detalle: Objetivo del proyecto, Análisis del dataset y la dimensionalidad (la versión final del EDA), Metodología (descripción detallada de las arquitecturas finales), Construcción de cada uno de los modelos, Entrenamiento e implementación (detalles del proceso, hiperparámetros, etc.), Resultados (todas las métricas reportadas, tablas y gráficos), Conclusiones (respondiendo a: ¿los resultados permiten un adecuado análisis y toma de decisiones para el problema estudiado?)."*

**Diferencia con el artículo IEEE:**
- **Sin límite de páginas** (puede ser de 20–40 páginas).
- **Mucho más detallado:** todas las gráficas, todas las tablas, todos los hiperparámetros explícitos.
- **Formato libre** (Word o LaTeX).
- **Audiencia:** alguien que quisiera reproducir el trabajo desde cero.

**Estructura sugerida:**

1. **Portada** (título, equipo, fecha).
2. **Resumen ejecutivo** (1 página).
3. **Objetivo del proyecto** (1 página).
4. **EDA completo** (4–8 páginas con todas las gráficas).
5. **Metodología** (descripción detallada de cada arquitectura con diagramas).
6. **Construcción de cada modelo** (subsección por modelo: arquitectura, hiperparámetros en tabla, decisiones de diseño).
7. **Entrenamiento e implementación** (procesos, callbacks, criterios de parada, manejo del desbalance).
8. **Resultados** (tabla comparativa, matrices de confusión, curvas, análisis por clase).
9. **Conclusiones** (responde explícitamente la pregunta del enunciado).
10. **Anexos** (código relevante, configuraciones completas).

**Tareas:**

- [ ] Crear estructura del documento en Word o LaTeX.
- [ ] Sección extensa de EDA con todas las gráficas (más detallado que en el artículo IEEE).
- [ ] Metodología: descripción detallada de cada arquitectura (texto + diagrama).
- [ ] Hiperparámetros completos en tabla por modelo.
- [ ] Resultados completos: matrices de confusión, métricas por clase, curvas de entrenamiento.
- [ ] Conclusiones respondiendo explícitamente: *¿los resultados permiten tomar decisiones para el problema?*

### 7.7 Lo que necesitas de los demás

- **De Felipe:** el resumen del EDA de dimensionalidad y clases (le sirve para decidir parámetros del pipeline).
- **De Daniel:** el esqueleto de entrenamiento, párrafo descriptivo de su modelo nuevo/combinado, hiperparámetros completos.
- **De Sebastián:** sus métricas y descripción del transformer (van como referencia en el reporte).

### 7.8 Lo que los demás necesitan de ti

- **Splits y pipeline de preprocesamiento** al final del viernes 8 (¡crítico!).
- **Métricas y figuras del modelo BiLSTM** al final del jueves 14.
- **Borrador del reporte técnico** el jueves 21.

### 7.9 Prompts útiles para tu LLM

- *"Aquí está la descripción del dataset y mi sección del plan. Diseña una función Python que haga preprocesamiento de texto en español: lowercase, manejo de tildes (preservar), remoción de URLs y emails, tokenización por palabras con vocabulario top-N."*
- *"Tengo un dataset desbalanceado de reseñas con 5 clases con esta distribución: [...]. ¿Qué estrategias de manejo de desbalance recomiendas y cómo las implemento en PyTorch?"*
- *"Revisa esta tabla de hiperparámetros de mi reporte técnico. ¿Está completa? ¿Falta algo que un lector necesitaría para reproducir el experimento?"*

---

## 8. Daniel Ruiz — Esqueleto + Modelo Nuevo/Combinado + Presentación

**Track:** Proyecto Principal
**Modelo a tu cargo:** Modelo Nuevo/Combinado (CNN+BiLSTM, BiLSTM+atención, etc.)
**Documento que coordinas:** Presentación de slides

### 8.1 Tu rol en el equipo

Eres responsable de **tres cosas grandes**:

1. **El esqueleto de entrenamiento común**, que reutilizan Felipe, Yibby y Sebastián. Si esto se hace bien, le ahorras horas a todo el equipo y garantiza que las métricas sean comparables.
2. **El modelo más complejo del proyecto principal**, donde el profesor evaluará la "profundidad técnica" más fuerte (ver criterio del enunciado).
3. **La presentación**, que es la cara visible del trabajo el día 25.

### 8.2 Esqueleto de entrenamiento — TAREA CRÍTICA (Lun 4 – Vie 8)

**Esto es lo primero que haces. Lo necesitan Felipe, Yibby y, en cierta medida, también te ayuda a ti y a Sebastián.**

**API sugerida en `src/training.py`:**

```python
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cuda"
) -> dict:
    """
    Entrena el modelo y retorna un dict con todas las métricas e historiales.
    
    config debe incluir:
        - epochs (int)
        - lr (float)
        - patience (int): para early stopping
        - class_weights (Tensor o None)
        - checkpoint_path (str): dónde guardar el mejor modelo
    
    Retorna el dict en formato JSON estandarizado del equipo.
    """
    pass

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """Evalúa en test set y retorna metrics dict."""
    pass
```

**Funcionalidades obligatorias:**

- [ ] Loop de entrenamiento estándar (forward, backward, step).
- [ ] **Early stopping:** detener si val loss no mejora en N épocas (paciencia 3–5).
- [ ] **Checkpoint del mejor modelo** (solo guardar cuando val loss mejora).
- [ ] **Logging por época:** train loss, val loss, train acc, val acc — guardados en historiales.
- [ ] **Class weights opcionales:** si están en config, pasar a `nn.CrossEntropyLoss(weight=...)`.
- [ ] **Learning rate scheduler opcional** (ReduceLROnPlateau).

**Funcionalidades en `src/metrics.py`:**

- [ ] `compute_metrics(y_true, y_pred) → dict`: accuracy, precision/recall/F1 macro y por clase, matriz de confusión.
- [ ] `save_metrics_json(metrics_dict, model_name, owner, track, config, training_info, path)`: guarda en formato del equipo.
- [ ] `plot_curves(loss_history, val_loss_history, acc_history, val_acc_history, save_path)`: PNG con 2 subplots.
- [ ] `plot_confusion_matrix(cm, class_names, save_path)`: PNG con matriz visual (matplotlib o seaborn).

**Entregable:** un README en `src/` que muestre cómo usar el esqueleto en menos de 10 líneas de código.

### 8.3 Modelo Nuevo/Combinado (Sáb 9 – Jue 14)

> Cita de `Proyecto_2026-2.pdf`, sección 3:
>
> *"Modelo nuevo/ combinado (que combine técnicas o use preprocesamiento avanzado, no es permitido el uso de redes pre-entrenadas)."*
>
> *"Evaluación: Se evaluará la Profundidad Técnica (¿Qué tan desafiante es el modelo?) y el Alcance (¿Cuántas variaciones o aspectos fueron explorados?)."*

**Esta es la sección donde se evalúa "profundidad técnica" más fuerte. La elección del modelo importa.**

**Opciones para considerar:**

| Opción | Pros | Contras |
|---|---|---|
| **CNN + BiLSTM** | CNN extrae n-gramas locales, BiLSTM captura dependencias largas. Combinación clásica robusta. | Más parámetros, entrenamiento más lento. |
| **BiLSTM + Atención** | El mecanismo de atención da interpretabilidad y suele mejorar resultados en clasificación de texto. | Implementación más sutil. |
| **Hierarchical Attention Network (HAN)** | Atención a nivel de palabra y de oración. Muy elegante para reseñas largas. | Requiere segmentar oraciones; más complejo. |
| **BiLSTM con embeddings pre-cargados FastText** | "Preprocesamiento avanzado" del enunciado. | El profesor podría considerar FastText como "pre-entrenado" — confirmar. |

**Recomendación:** **BiLSTM + Atención**. Balance bueno entre profundidad técnica y factibilidad en el tiempo disponible. Permite mostrar interpretabilidad (visualizar a qué palabras presta atención el modelo), que queda muy bien en el reporte y la presentación.

**Tareas:**

- [ ] Investigar 2–3 papers o tutoriales de la arquitectura elegida.
- [ ] Justificar técnicamente la elección en el README del notebook (esto es central para la calificación).
- [ ] Implementar la arquitectura **respetando la prohibición: nada de redes pre-entrenadas**.
- [ ] Iterar 2–3 configuraciones de hiperparámetros.
- [ ] Si eliges atención: generar visualización de pesos de atención sobre 3–5 reseñas representativas (queda muy bien en la presentación).
- [ ] Generar:
  - `results/combined_metrics.json`
  - `figures/combined_curves.png`
  - `figures/combined_confusion.png`
  - `figures/combined_attention_examples.png` (si aplica)
  - `results/combined_best.pt`
- [ ] Escribir un párrafo descriptivo de la arquitectura (2–3 oraciones) y compartirlo con Felipe y Yibby.

### 8.4 Presentación (Vie 15 – Jue 21)

> Cita de `Proyecto_2026-2.pdf`, sección 7:
>
> *"Al finalizar el curso, se realizará una presentación del proyecto. Formato: Se sugiere la elaboración de material digital (diapositivas)."*

**Estructura sugerida (12–15 minutos + Q&A):**

| Slide(s) | Contenido | Quién presenta |
|---|---|---|
| 1 | Portada con título y equipo | Daniel |
| 2 | Contexto y problema | Felipe (es quien escribió la intro del artículo) |
| 3 | Objetivos | Felipe |
| 4–5 | Dataset y EDA (gráficas clave: distribución de clases, longitudes) | Yibby (ella lideró el EDA de features) |
| 6 | Pipeline de preprocesamiento | Yibby |
| 7 | Modelo Clásico 1 (LSTM) — diagrama y métricas | Felipe |
| 8 | Modelo Clásico 2 (BiLSTM) — diagrama y métricas | Yibby |
| 9 | Modelo Nuevo/Combinado — diagrama, justificación, métricas | Daniel |
| 10 | Modelo Avanzado (Transformer) — diagrama, métricas | Sebastián |
| 11 | **Tabla comparativa** de los 4 modelos (clave) | Daniel |
| 12 | Análisis crítico (qué funcionó, qué no, por qué) | Daniel |
| 13 | Conclusiones | Daniel |
| 14 | Trabajo futuro / preguntas | Todos |

**Decisiones a tomar:**
- Herramienta: PowerPoint, Google Slides, Keynote, o LaTeX Beamer.
- Plantilla visual coherente.
- Tipografía legible desde el fondo del salón.

**Tareas:**

- [ ] Crear plantilla y estructura de slides.
- [ ] Recopilar las gráficas que cada uno generó (ya están en `figures/`).
- [ ] Diseñar diagramas visuales de cada arquitectura (puedes usar diagrams.net, draw.io, o `nn-svg`).
- [ ] Redactar el guion de tu parte (modelo nuevo, comparación, conclusiones).
- [ ] **Coordinar el ensayo del fin de semana 22–24 con cronómetro**; cada uno presenta su parte.
- [ ] Asegurar que los nombres y créditos de los 4 estén visibles.

### 8.5 Lo que necesitas de los demás

- **De Yibby:** los splits y el pipeline para entrenar tu modelo, su descripción de modelo BiLSTM y figuras.
- **De Felipe:** descripción de modelo LSTM y figuras, descripción del problema y EDA.
- **De Sebastián:** descripción del transformer, figuras, y métricas para la tabla comparativa.
- **De todos:** disponibilidad para el ensayo el sábado 23 y domingo 24.

### 8.6 Lo que los demás necesitan de ti

- **Esqueleto de entrenamiento funcional** al final del viernes 8 (¡crítico!).
- **Métricas y figuras del modelo nuevo/combinado** al final del jueves 14.
- **Slides completos** el jueves 21.
- **Coordinación del ensayo** sábado 23.

### 8.7 Prompts útiles para tu LLM

- *"Implementa en PyTorch un BiLSTM con mecanismo de atención (Bahdanau o auto-atención simple) para clasificación de texto multi-clase. Incluye comentarios explicando cada paso."*
- *"Genera un diagrama en código (Mermaid o gráfico Python) de la siguiente arquitectura: Embedding → BiLSTM → Atención → Dense → Softmax. Indícame también qué herramientas visuales podría usar para mejorarlo."*
- *"Tengo 12 minutos para presentar 4 modelos de Deep Learning a una audiencia técnica. ¿Cómo distribuyo el tiempo y cuáles son las 3 ideas clave que no puedo dejar de mencionar por modelo?"*

---

## 9. Sebastián Ruiz — Modelo Avanzado (Transformer) + Informe de investigación

**Track:** Tarea de Investigación (entrega separada del Proyecto Principal)
**Modelo a tu cargo:** Fine-tuning de Transformer pre-entrenado en español
**Documento que coordinas:** Informe de investigación (máx. 5 páginas)

### 9.1 Tu rol en el equipo

Tu trabajo es **una entrega separada** del Proyecto Principal:

> Cita de `Proyecto_2026-3.pdf`:
>
> *"Esta fase es una tarea de investigación técnica que complementa el proyecto principal."*

Eres responsable de **dos cosas**:

1. **Fine-tuning de un Transformer pre-entrenado** (BETO, BERTIN o similar) — distinto al modelo nuevo/combinado de Daniel, que NO puede usar redes pre-entrenadas.
2. **El informe de investigación**, un PDF separado del artículo IEEE y del reporte técnico (máximo 5 páginas).

**Tu trabajo es independiente del pipeline de preprocesamiento de Yibby**: el transformer trae su propio tokenizador. Esto te permite **arrancar desde el día 2 sin esperar a nadie**.

### 9.2 Setup independiente (Lun 4 – Vie 8)

**Esto es paralelo al trabajo del resto. No esperes a Yibby ni a Daniel — su trabajo no afecta el tuyo (excepto los splits, que cargas en lugar de regenerar).**

**Tareas:**

- [ ] Validar que tienes GPU en Colab (`Runtime → Change runtime type → GPU`).
- [ ] Instalar HuggingFace:
  ```
  pip install transformers accelerate datasets
  ```
- [ ] **Decidir el modelo base.** Opciones:

| Modelo | HuggingFace ID | Notas |
|---|---|---|
| **BETO** | `dccuchile/bert-base-spanish-wwm-uncased` | El más usado para español, robusto. Tu primera opción. |
| **BETO cased** | `dccuchile/bert-base-spanish-wwm-cased` | Si las mayúsculas importan en tu corpus. |
| **BERTIN** | `bertin-project/bertin-roberta-base-spanish` | RoBERTa entrenada en español. Alternativa moderna. |
| **MarIA** | `PlanTL-GOB-ES/roberta-base-bne` | Del gobierno español, entrenada en BNE. |

- [ ] Documentar en el README la decisión (probablemente BETO; justifica con 2–3 líneas).
- [ ] Cargar tokenizer y modelo:
  ```python
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
  model = AutoModelForSequenceClassification.from_pretrained(
      "dccuchile/bert-base-spanish-wwm-uncased", num_labels=NUM_CLASSES
  )
  ```
- [ ] **Smoke test:** fine-tuning con 100 ejemplos durante 1 época. Si esto corre sin error en GPU, el resto es escalado.
- [ ] Validar que el tokenizer produce salidas razonables sobre 5–10 reseñas reales.

**Cuando estén los splits de Yibby (probablemente jueves 7 o viernes 8):**
- [ ] Cargarlos desde `data/train.csv`, `val.csv`, `test.csv`.
- [ ] Aplicar el tokenizer del transformer (NO el pipeline de Yibby — esto es importante: el transformer tiene su propio vocabulario y subword tokenization).

### 9.3 Fine-tuning del transformer (Sáb 9 – Jue 14)

> Cita de `Proyecto_2026-3.pdf`, sección "Tarea Específica para el Enfoque 2", apartado B:
>
> *"Implementar un modelo usando Fine-Tuning de un Transformer. Detalle: Deberán usar un modelo de lenguaje pre-entrenado (ej. BETO, RoBERTa) y realizar fine-tuning de (al menos) la última capa del transformer para la tarea de clasificación."*

**Decisiones a tomar:**

- [ ] **Capas a descongelar.** El enunciado pide *al menos la última capa*. Recomendación: explorar 2 configuraciones:
  1. Solo la cabeza de clasificación + última capa encoder (más rápido, menos riesgo de overfitting).
  2. Cabeza + último bloque completo (4 capas si es BERT-base) (más capacidad, más riesgo de overfitting).
- [ ] **Hiperparámetros típicos para fine-tuning de transformer:**
  - `learning_rate`: 2e-5 a 5e-5 (¡muy bajo! es la diferencia clave con los modelos clásicos).
  - `batch_size`: 16 o 32.
  - `epochs`: 3 a 5 (transformers hacen overfitting muy rápido).
  - `warmup`: 10% de los steps totales.
  - `weight_decay`: 0.01.
  - `max_length`: 128 o 256 tokens (depende del percentil 95 de Felipe).

**Implementación con HuggingFace Trainer:**

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results/transformer",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
```

**Tareas:**

- [ ] Implementar el fine-tuning (con Trainer o con loop manual).
- [ ] Implementar `compute_metrics_fn` que retorne accuracy, F1 macro y por clase.
- [ ] Monitorear overfitting: el val loss sube rápido en transformers — early stopping con paciencia 1 o 2.
- [ ] Guardar el mejor checkpoint y métricas finales.
- [ ] Generar:
  - `results/transformer_metrics.json`
  - `figures/transformer_curves.png` (curvas obligatorias en el informe).
  - `figures/transformer_confusion.png`
  - `results/transformer_best/` (carpeta con checkpoint).

### 9.4 Informe de investigación (Vie 15 – Jue 21)

> Cita de `Proyecto_2026-3.pdf`, sección "Entregables de la Fase 3":
>
> *"Informe de Investigación (PDF) (Máx. 5 páginas). Un reporte técnico separado que detalle exclusivamente esta tarea de investigación. Introducción: Breve descripción del objetivo (ej. 'Evaluar el impacto de Fine-Tuning...'). Metodología: Descripción de la arquitectura avanzada implementada (qué modelo base se usó, qué capas se descongelaron, qué hiperparámetros se usaron). Resultados: Una tabla comparativa que enfrente las métricas de este modelo avanzado contra los dos modelos clásicos (cuyos resultados se obtuvieron en la Fase 2). Gráficos de las curvas de pérdida (Loss) y precisión (Accuracy) de entrenamiento vs. validación del modelo avanzado. Análisis y Conclusiones: ¿El modelo avanzado superó a los modelos clásicos? ¿Por qué? Análisis de la profundidad técnica: ¿Qué tan desafiante fue la implementación, el costo computacional y el ajuste de hiperparámetros?"*

**Estructura obligatoria** (máx. 5 páginas):

| Sección | Páginas |
|---|---|
| 1. Introducción | 0.5 |
| 2. Metodología | 1 |
| 3. Resultados (tabla + curvas) | 1.5 |
| 4. Análisis y conclusiones | 1.5 |
| Referencias | 0.5 |

**Tabla comparativa obligatoria** (esto es lo más importante del informe):

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Parámetros | Tiempo entrenamiento |
|---|---|---|---|---|---|---|
| LSTM (Clásico 1, Felipe) | ... | ... | ... | ... | ... | ... |
| BiLSTM (Clásico 2, Yibby) | ... | ... | ... | ... | ... | ... |
| BETO fine-tuned (Avanzado) | ... | ... | ... | ... | ... | ... |

**⚠️ Importante:** la tabla compara contra los **2 modelos clásicos**, NO contra el modelo nuevo/combinado de Daniel. Ese pertenece al Proyecto Principal, no a tu Tarea de Investigación.

**Análisis y conclusiones — preguntas obligatorias a responder:**

- [ ] ¿El modelo avanzado superó a los modelos clásicos? Cuantificar (¿en cuántos puntos de F1 macro?).
- [ ] Si sí: ¿por qué? (capacidad del modelo, embeddings contextuales, conocimiento pre-entrenado del español).
- [ ] Si no: ¿por qué? (dataset pequeño, dominio muy específico, overfitting).
- [ ] ¿Qué tan desafiante fue la implementación? (versiones de librerías, GPU, etc.).
- [ ] ¿Cuál fue el costo computacional? (tiempo, memoria, comparado con los clásicos).
- [ ] ¿Cómo fue el ajuste de hiperparámetros? (qué probaste, qué funcionó).

**Tareas:**

- [ ] Crear estructura del documento (LaTeX o Word).
- [ ] Redactar metodología (puede arrancar el viernes 15 sin esperar resultados finales).
- [ ] Generar tabla comparativa con los 3 modelos.
- [ ] Incluir las curvas loss/accuracy del transformer (obligatorias).
- [ ] Redactar análisis crítico respondiendo todas las preguntas anteriores.
- [ ] Limitar a 5 páginas exactas.

### 9.5 Lo que necesitas de los demás

- **De Yibby:** los splits guardados como archivos (no necesitas el pipeline porque tu tokenizer es del transformer).
- **De Felipe y Yibby:** sus métricas en JSON estandarizado para tu tabla comparativa.
- **Identificación del mejor clásico** (entre LSTM de Felipe y BiLSTM de Yibby) — decisión grupal del jueves 14.

### 9.6 Lo que los demás necesitan de ti

- **Tus métricas del transformer** al final del jueves 14 (van en el artículo IEEE, en el reporte técnico y en la presentación, aunque la entrega formal de tu trabajo sea por separado).
- **Párrafo descriptivo del transformer** para que Felipe, Yibby y Daniel lo integren en sus documentos.
- **Borrador del informe** el jueves 21.

### 9.7 Diferencias clave con el resto del equipo

| Aspecto | Tú (Sebastián) | Felipe / Yibby / Daniel |
|---|---|---|
| Track | Tarea de Investigación | Proyecto Principal |
| Restricción | DEBE usar pre-entrenado | NO PUEDE usar pre-entrenado (en el modelo nuevo) |
| Tokenizer | Del transformer (BETO) | El de Yibby (palabras + padding) |
| Documento propio | Informe de investigación (5 pág) | Artículo IEEE / Reporte / Presentación |
| Compara contra | 2 modelos clásicos | (no aplica) |

### 9.8 Prompts útiles para tu LLM

- *"Implementa fine-tuning de BETO para clasificación de reseñas en español multi-clase usando HuggingFace Trainer. El dataset está desbalanceado, así que aplica class weights en la loss. Incluye early stopping y eval por época."*
- *"¿Cuál es la diferencia práctica entre descongelar solo la última capa del encoder vs. el último bloque completo en BERT? ¿Cuándo conviene cada uno?"*
- *"Tengo este resultado: BETO fine-tuned obtuvo F1 macro 0.87 vs. BiLSTM 0.81. Ayúdame a redactar un análisis de 200 palabras que explique por qué y discuta los trade-offs computacionales."*

---

# Parte III — Hitos y validación

## 10. Hito 1 — Viernes 8 de mayo · Cimientos listos

**Antes del viernes en la noche, el equipo se reúne (30 min) y valida que todos los siguientes ítems estén marcados. Si alguno falta, se discute si se posterga o se reduce alcance.**

- [ ] Repositorio Git creado, los 4 tienen acceso de escritura. *(Yibby)*
- [ ] `data/train.csv`, `data/val.csv`, `data/test.csv` en el repo, con semilla 42 y splits estratificados. *(Yibby)*
- [ ] `src/preprocessing.py` empaquetado con `build_pipeline()`. *(Yibby)*
- [ ] `src/training.py` y `src/metrics.py` con esqueleto reusable + README de uso. *(Daniel)*
- [ ] Notebooks de EDA (`01_eda_felipe.ipynb` y `02_eda_yibby.ipynb`) ejecutados de extremo a extremo, con hallazgos clave documentados. *(Felipe + Yibby)*
- [ ] Smoke test del transformer corriendo en GPU: 100 ejemplos, 1 época, sin error. *(Sebastián)*
- [ ] Decisión documentada en el README: qué modelo base de transformer se usa, qué embeddings para los clásicos, qué long. máxima de secuencia.

## 11. Hito 2 — Jueves 14 de mayo · Modelos congelados

**Reunión de 30 min en la noche. Después de este punto, NO MÁS EXPERIMENTOS — la documentación necesita números estables.**

- [ ] `results/lstm_metrics.json` con métricas finales del Modelo Clásico 1. *(Felipe)*
- [ ] `results/bilstm_metrics.json` con métricas finales del Modelo Clásico 2. *(Yibby)*
- [ ] `results/combined_metrics.json` con métricas finales del Modelo Nuevo/Combinado. *(Daniel)*
- [ ] `results/transformer_metrics.json` con métricas finales del Modelo Avanzado. *(Sebastián)*
- [ ] Checkpoint del mejor modelo guardado para cada uno (`results/{modelo}_best.pt` o equivalente).
- [ ] Curvas loss/accuracy (train vs. val) exportadas como PNG en `figures/` para los 4 modelos.
- [ ] Matrices de confusión exportadas como PNG en `figures/` para los 4 modelos.
- [ ] **Decisión grupal documentada: cuál es el mejor modelo clásico** (entre el de Felipe y el de Yibby) — base de comparación para Sebastián.
- [ ] Revisión grupal: ¿hay anomalías?, ¿algún modelo no entrenó bien?, ¿hay que reentrenar algo antes del lockdown?

## 12. Hito 3 — Jueves 21 de mayo · Borradores 1.0 completos

**Después de este punto: solo revisión y pulido, no más cambios estructurales.**

- [ ] **[PP]** Artículo IEEE en formato Conference: 4–6 páginas, todas las secciones, figuras numeradas, ≥ 8 referencias. *(Felipe)*
- [ ] **[PP]** Reporte técnico completo: EDA extendido, metodología detallada, todos los hiperparámetros en tablas, conclusiones. *(Yibby)*
- [ ] **[PP]** Presentación con todas las slides: contexto, EDA, 4 arquitecturas, comparación, conclusiones. *(Daniel)*
- [ ] **[TI]** Informe de investigación: máx. 5 páginas, tabla comparativa transformer vs. 2 clásicos, curvas loss/accuracy, análisis. *(Sebastián)*
- [ ] Tabla comparativa final con los 4 modelos consensuada y replicada en todos los documentos donde aplique.
- [ ] Cada documento revisado por su autor al menos una vez antes de pasarlo a revisión cruzada.

## 13. Días finales (Vie 22 – Lun 25)

- **Vie 22 – Sáb 23:** revisión cruzada (cada uno revisa el documento que NO coordinó). Ensayo de presentación con cronómetro. Sábado noche: PDFs finales subidos al repo.
- **Dom 24:** día buffer para imprevistos. Ensayo final.
- **Lun 25:** entrega y presentación.

---

# Parte IV — Apéndices y errores típicos

## 14. Errores típicos a evitar

1. **Reportar solo accuracy.** El dataset está desbalanceado. Una accuracy alta puede ser engañosa (un modelo trivial que predice la clase mayoritaria puede tener accuracy alta). **Reportar siempre F1 macro y por clase.**

2. **Tunear hiperparámetros mirando el test set.** Esto es *data leakage*. El test set se mira **una sola vez por modelo**, al final, para reportar la métrica final. Para tunear, se usa el val set.

3. **Splits inconsistentes entre miembros.** Si cada uno hace su propio split, las métricas no son comparables y la tabla comparativa final se cae. **Yibby genera, todos cargan.**

4. **Olvidar guardar los modelos.** Guardar solo métricas no permite re-evaluar después. Siempre `torch.save(model.state_dict(), ...)` o equivalente.

5. **Transformer entrenado demasiadas épocas.** Los transformers fine-tuned hacen overfitting muy rápido. 3–5 épocas máximo, early stopping agresivo (paciencia 1 o 2).

6. **Subestimar el tiempo de redacción.** Los documentos son una parte enorme de la calificación. Por eso el hito 3 es 4 días antes de la entrega: hay que dejar tiempo para revisar, no solo escribir.

7. **Reproducir el modelo nuevo/combinado en la Tarea de Investigación.** Son entregas separadas. La Tarea de Investigación compara solo contra los 2 clásicos.

8. **No usar el pipeline común.** Si Felipe o Yibby modifican el preprocesamiento "solo un poquito" para su modelo, las comparaciones ya no son justas. **Si necesitas modificar el pipeline, lo discutes con el grupo y se aplica a todos.**

9. **Hiperparámetros no documentados.** Si en el reporte técnico la tabla de hiperparámetros está incompleta, es un golpe directo a la calificación de "alcance".

10. **Confundir el orden de las clases.** Si la matriz de confusión usa un orden y el reporte por clase usa otro, las interpretaciones están mal. **Acordar un orden de clases único en el `metrics.py` de Daniel.**

## 15. Apéndice técnico — Snippets reusables

### 15.1 Carga de splits estandarizada

```python
import pandas as pd
import numpy as np
import random
import torch

# Semilla
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Carga
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print("Distribución de clases en train:", train["label"].value_counts(normalize=True))
```

### 15.2 Class weights para dataset desbalanceado

```python
from sklearn.utils.class_weight import compute_class_weight
import torch

classes = np.unique(train["label"])
weights = compute_class_weight("balanced", classes=classes, y=train["label"])
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
```

### 15.3 Métricas estandarizadas

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_per_class": {str(i): float(v) for i, v in enumerate(f1_per_class)},
        "confusion_matrix": cm.tolist(),
    }
```

### 15.4 Tabla comparativa final (script)

```python
import json
from pathlib import Path
import pandas as pd

results_dir = Path("results")
all_metrics = []
for json_file in results_dir.glob("*_metrics.json"):
    with open(json_file) as f:
        data = json.load(f)
    all_metrics.append({
        "Modelo": data["model_name"],
        "Owner": data["owner"],
        "Track": data["track"],
        "Accuracy": data["metrics"]["accuracy"],
        "F1 macro": data["metrics"]["f1_macro"],
        "Precision macro": data["metrics"]["precision_macro"],
        "Recall macro": data["metrics"]["recall_macro"],
        "Parámetros": data["config"].get("n_params", "N/A"),
        "Tiempo (s)": data["training"]["training_time_seconds"],
    })

comparison_df = pd.DataFrame(all_metrics)
comparison_df.to_csv("results/comparison_table.csv", index=False)
print(comparison_df.to_markdown(index=False))
```

## 16. FAQ

**P: ¿Puedo usar TensorFlow/Keras en lugar de PyTorch?**
R: Sí, pero acuerden todo el equipo el mismo framework. Mezclar es fuente de bugs y de inconsistencias de métricas. Sugerencia: PyTorch (es lo que mejor soporta HuggingFace para Sebastián).

**P: ¿Y si el dataset que descargo no tiene las columnas que esperamos?**
R: Lo descubre Felipe en la primera inspección del lunes 4. En la reunión de kickoff del martes, se ajusta el plan en función de lo que el dataset realmente tenga.

**P: ¿Cuento o no cuento las stop words?**
R: En análisis de sentimiento en español, palabras como "no", "muy", "poco", "nada" son cruciales. Recomendación: **NO removerlas**. Documentar la decisión en el reporte.

**P: ¿Mi modelo tiene F1 macro 0.50 — ¿está mal?**
R: Depende del baseline. Si el dataset tiene 5 clases balanceadas, predecir aleatorio da F1 = 0.20. Si está desbalanceado, predecir siempre la clase mayoritaria puede dar accuracy alta pero F1 macro baja. Compara contra esos baselines triviales primero.

**P: ¿Puedo usar embeddings pre-entrenados (FastText, Word2Vec) en el modelo nuevo/combinado?**
R: El enunciado prohíbe "redes pre-entrenadas". FastText/Word2Vec son embeddings, no redes; suelen aceptarse. Pero **confirma con el profesor** si tienes dudas — la decisión segura es usar embeddings entrenables desde cero.

**P: ¿Cuándo le pregunto al profesor?**
R: Para dudas de alcance (qué cuenta como "nuevo modelo", si vale tal arquitectura). Para dudas técnicas (cómo implementar atención), tu LLM y este documento son suficientes.

---

## 17. Checklist final del lunes 25

Antes de salir de casa el día de la entrega:

- [ ] Repositorio Git con todo el código, limpio, con README que explica cómo reproducir.
- [ ] Artículo IEEE en PDF (4–6 páginas).
- [ ] Reporte técnico en PDF.
- [ ] Notebook(s) ejecutados con salidas guardadas.
- [ ] Informe de investigación en PDF (5 páginas máx).
- [ ] Presentación en PDF + archivo editable (.pptx o .key).
- [ ] Checkpoints de los 4 modelos en el repo (o link a Drive si pesan demasiado).
- [ ] Los 4 nombres y roles claros en cada documento.

**¡Suerte equipo!** 🚀

---

*Documento generado el 7 de mayo de 2026 a partir de los enunciados oficiales `Proyecto_2026-2.pdf` (Proyecto Principal) y `Proyecto_2026-3.pdf` (Tarea de Investigación) del profesor Ing. Julio Omar Palacio Niño, M.Sc.*
