# Entregables - Proyecto Deep Learning (RNN)

**Asignatura:** Aprendizaje Profundo - Maestria en IA
**Universidad:** Pontificia Universidad Javeriana, Bogota
**Profesor:** Ing. Julio Omar Palacio Nino, M.Sc.
**Fecha de entrega:** Lunes 25 de mayo de 2026
**Equipo:** Felipe Reyes, Yibby Gonzalez, Daniel Ruiz, Sebastian Ruiz

---

## Estructura de entrega

```
entregables/
|
|-- 01_Proyecto_Principal/          (Felipe, Yibby, Daniel)
|   |-- 01_Articulo_IEEE/           Articulo formato IEEE Conference (4-6 pag, PDF)
|   |-- 02_Reporte_Tecnico/         Reporte tecnico detallado (PDF, sin limite pag)
|   |-- 03_Presentacion/            Diapositivas (PDF + editable)
|   |-- 04_Notebooks/               5 notebooks ejecutados con salidas
|   |   |-- 01_eda_felipe.ipynb
|   |   |-- 02_eda_yibby.ipynb
|   |   |-- 03_lstm_felipe.ipynb
|   |   |-- 04_bilstm_yibby.ipynb
|   |   |-- 05_combined_daniel.ipynb
|   |-- 05_Codigo_Fuente/            Modulos Python reutilizables
|   |   |-- preprocessing.py         Pipeline de preprocesamiento (Yibby)
|   |   |-- training.py              Esqueleto de entrenamiento (Daniel)
|   |   |-- metrics.py               Calculo y visualizacion de metricas (Daniel)
|   |-- 06_Resultados/               Metricas JSON + tabla comparativa
|   |   |-- lstm_metrics.json
|   |   |-- bilstm_metrics.json
|   |   |-- bilstm_multihead_metrics.json
|   |   |-- tabla_comparativa.csv
|   |-- 07_Figuras/                  Graficas organizadas por seccion
|       |-- EDA/                     8 figuras de analisis exploratorio
|       |-- LSTM/                    Curvas y matriz de confusion (v3 definitivo)
|       |-- BiLSTM/                  Curvas y matriz de confusion (v1 definitivo)
|       |-- Modelo_Combinado/        Curvas, confusion y comparacion F1 por clase
|
|-- 02_Tarea_Investigacion/         (Sebastian)
    |-- 01_Informe_Investigacion/    Informe tecnico (max 5 pag, PDF)
    |-- 02_Notebook/                 Notebook del transformer ejecutado
    |   |-- 06_transformer_sebas.ipynb
    |-- 03_Resultados/               Metricas JSON del transformer
    |   |-- transformer_metrics.json
    |-- 04_Figuras/                  Curvas y matrices de confusion
```

---

## Checklist final de entrega

### Proyecto Principal (PP)
- [ ] Articulo IEEE en PDF (4-6 paginas, formato IEEE Conference) -- Felipe
- [ ] Reporte tecnico en PDF -- Yibby
- [ ] Presentacion en PDF + archivo editable (.pptx/.key) -- Daniel
- [x] Notebooks 01-05 ejecutados con salidas guardadas
- [x] Codigo fuente (preprocessing.py, training.py, metrics.py)
- [x] Metricas JSON de los 3 modelos PP
- [x] Tabla comparativa CSV
- [x] Figuras EDA (8 graficas)
- [x] Figuras LSTM v3 definitivo (curvas + confusion)
- [x] Figuras BiLSTM v1 definitivo (curvas + confusion)
- [x] Figuras Modelo Combinado (curvas + confusion + F1 por clase)

### Tarea de Investigacion (TI)
- [ ] Informe de investigacion en PDF (max 5 paginas) -- Sebastian
- [ ] **Regenerar figuras de v3** (las actuales son de v1, ver nota abajo)
- [x] Notebook 06 ejecutado con salidas
- [x] Metricas JSON del transformer

---

## NOTA IMPORTANTE: Figuras del Transformer

Las figuras actuales en `02_Tarea_Investigacion/04_Figuras/` corresponden
a la **version v1** del transformer (baseline). El modelo definitivo es **v3**.

Los datos para regenerar las curvas y matriz de confusion de v3 estan en:
`docs/originales/fuente_tarea_investigacion.md` (secciones 4 y 9).

---

## Resumen de modelos y metricas finales

| Modelo | Track | F1 macro | Accuracy | Responsable |
|---|---|---|---|---|
| LSTM | PP | 0.3996 | 0.6042 | Felipe |
| BiLSTM | PP | 0.5749 | 0.6871 | Yibby |
| BiLSTM+MultiHead | PP | 0.6193 | 0.6548 | Daniel |
| BETO v3 fine-tuned | TI | 0.6565 | 0.7252 | Sebastian |
