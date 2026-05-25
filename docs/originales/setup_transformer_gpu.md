# Setup: Notebook Transformer (BETO) en equipo con GPU

Guia paso a paso para instalar Miniconda desde cero y ejecutar el notebook
`06_transformer_sebas.ipynb` en un equipo local con GPU NVIDIA.

---

## Requisitos previos

- Equipo con GPU NVIDIA (drivers instalados)
- Sistema operativo: Windows 10/11 o Linux (Ubuntu 20.04+)
- Conexion a internet (para descargar paquetes y el modelo BETO de HuggingFace)

Para verificar que los drivers NVIDIA estan instalados:

```bash
nvidia-smi
```

Debe mostrar la GPU y la version del driver. Si no funciona, instalar drivers desde
https://www.nvidia.com/Download/index.aspx antes de continuar.

---

## Paso 1: Instalar Miniconda

### Windows

1. Descargar el instalador desde https://docs.anaconda.com/miniconda/
2. Ejecutar el `.exe` y seguir el asistente (marcar "Add to PATH" si se desea)
3. Abrir **Anaconda Prompt** (o reiniciar la terminal)

### Linux

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
```

Cerrar y reabrir la terminal.

### Verificar instalacion

```bash
conda --version
```

---

## Paso 2: Crear el entorno con Python 3.11

```bash
conda create -n beto_ft python=3.11 -y
conda activate beto_ft
```

> Python 3.11 es la version mas estable y compatible con PyTorch + transformers
> a la fecha.

---

## Paso 3: Instalar PyTorch con soporte CUDA

Ir a https://pytorch.org/get-started/locally/ y seleccionar la combinacion
correcta. Para la mayoria de equipos recientes (CUDA 12.x):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> Si el equipo tiene un driver mas antiguo (CUDA 11.8), usar:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Verificar que PyTorch detecta la GPU

```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Debe imprimir `CUDA disponible: True` y el nombre de la GPU.
**Si dice False, NO continuar** — revisar drivers y version de CUDA.

---

## Paso 4: Instalar dependencias del notebook

```bash
pip install transformers accelerate scikit-learn pandas numpy matplotlib seaborn tqdm jupyter
```

Resumen de paquetes y su funcion:

| Paquete        | Para que se usa                              |
|----------------|----------------------------------------------|
| transformers   | Modelo BETO (HuggingFace)                    |
| accelerate     | Requerido por transformers para carga optima  |
| scikit-learn   | Class weights, metricas (F1, confusion matrix)|
| pandas         | Lectura de CSVs (train/val/test)             |
| numpy          | Operaciones numericas                        |
| matplotlib     | Graficas (curvas de entrenamiento)           |
| seaborn        | Heatmaps (matrices de confusion)             |
| tqdm           | Barras de progreso durante entrenamiento     |
| jupyter        | Ejecutar el notebook (.ipynb)                |

---

## Paso 5: Clonar el repositorio

```bash
git clone https://github.com/JuanSe1796/Proyecto_Deep_Learning_Final.git
cd Proyecto_Deep_Learning_Final/notebooks
```

Verificar que existen los datos:

```bash
ls ../data/train.csv ../data/val.csv ../data/test.csv
```

Si no existen los CSVs, pedirlos al equipo (Subset generado por Sebastian con semilla 42).

---

## Paso 6: Ejecutar el notebook

### Opcion A: Jupyter Notebook (interfaz web)

```bash
jupyter notebook 06_transformer_sebas.ipynb
```

Se abre en el navegador. Ejecutar celda por celda con `Shift+Enter`
o todo de corrido con **Kernel > Restart & Run All**.

### Opcion B: Desde terminal (sin interfaz)

```bash
jupyter nbconvert --to notebook --execute 06_transformer_sebas.ipynb --output 06_transformer_sebas_ejecutado.ipynb
```

> **Nota:** La celda 1 (Setup Colab) intenta clonar el repo en `/content/`,
> que es la ruta de Colab. En un equipo local esto no causa error si ya
> estas dentro de `notebooks/`, pero si falla, simplemente saltarla
> y ejecutar desde la celda 2 en adelante.
>
> La celda 26 (Descarga de archivos) usa `google.colab.files` que solo funciona
> en Colab. En local, simplemente ignorarla — los archivos ya quedan guardados
> en `results/` y `figures/`.

---

## Paso 7: Verificar resultados

Al terminar, deben existir estos archivos:

```
Proyecto_Deep_Learning_Final/
  results/
    transformer_best.pt              # Checkpoint del mejor modelo
    transformer_metrics.json         # Metricas en formato JSON
  figures/
    transformer_curves.png           # Curvas de loss y accuracy
    transformer_confusion.png        # Matriz de confusion (normalizada)
    transformer_confusion_absolutos.png  # Matriz de confusion (absoluta)
```

---

## Tiempo estimado de ejecucion

| GPU               | Tiempo aproximado (5 epocas) |
|--------------------|------------------------------|
| RTX 3060 / 3070    | ~10-15 min                   |
| RTX 3080 / 3090    | ~7-10 min                    |
| RTX 4070 / 4080    | ~5-8 min                     |
| RTX 4090            | ~4-5 min                     |
| T4 (Colab gratis)  | ~11 min                      |

---

## Troubleshooting

### `torch.cuda.is_available()` devuelve `False`

- Verificar que `nvidia-smi` funciona
- Verificar que se instalo la version correcta de PyTorch (cu124 vs cu118)
- En Windows, asegurar que no se instalo la version CPU por error

### `OutOfMemoryError: CUDA out of memory`

- Reducir `BATCH_SIZE` de 32 a 16 en la celda 4b
- GPUs con menos de 4GB de VRAM pueden tener problemas con BETO

### Error en la celda de Setup Colab

- Ignorar la celda 1 (Setup Colab) y ejecutar desde la celda 2
- Asegurarse de estar en la carpeta `notebooks/` del repositorio

### `ModuleNotFoundError: No module named 'transformers'`

- Verificar que el entorno conda esta activado: `conda activate beto_ft`
- Reinstalar: `pip install transformers`
