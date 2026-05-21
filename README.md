# Taller B3-T4 — Redes Neuronales para Forecasting

Práctica del bloque 3 del Máster MIAX. Aplicación de redes neuronales (densas, recurrentes, convolucionales y mixtas) al forecasting de retornos de 23 activos del S&P 500, más una cartera de inversión construida sobre las predicciones del mejor modelo.

**Autores:** Javier Hernández, Diego Muñoz, Javier Fernández
**Entrega:** 21 de mayo de 2026

---

## 1. Objetivo de la práctica

Estimar el promedio del log-retorno al cierre de los próximos `v_out` días para los 23 activos, a partir de una ventana de `v_in` días pasados. Los modelos se entrenan minimizando el **MAE** (Mean Absolute Error) y se reporta el error en *train*, *validación* y *test* para cada combinación de ventanas.

Se cubren cuatro familias de arquitecturas: capas densas (MLP), capas recurrentes (LSTM/GRU), capas convolucionales (Conv1D) y mixtas. Adicionalmente se realiza una **investigación** que aplica preprocesado financiero (diferenciación fraccional) y, con el mejor modelo a 90 días, se construyen dos carteras (con y sin predicciones) que se comparan sobre 2025.

## 2. Datos

- **Universo:** 23 tickers fijos del S&P 500 — `AEP, BA, CAT, CNP, CVX, DIS, DTE, ED, GD, GE, HON, HPQ, IBM, IP, JNJ, KO, KR, MMM, MO, MRK, MSI, PG, XOM`.
- **Histórico:** descarga diaria vía `yfinance` desde 1962-01-03 (~16.000 días).
- **Fichero de lectura proporcionado:** `00_datos/Lectura_datos_Taller_B3_T4.ipynb`.

Los CSV no se commitean: cada notebook descarga los datos por código.

## 3. Preprocesado

Implementado en `01_src_compartido/dataset_utils.py` y reutilizado por todos los notebooks.

- **Entrada `X`** — log-retornos diarios `log(close).diff()`, con shape `(N, v_in, 23)`.
- **Salida `Y`** — media de log-retornos durante los `v_out` días futuros, con shape `(N, 23)`.
- **Partición train/test 90/10** con `random_state=42` (semilla fija de competición).
- **Escalado opcional** `standard` o `minmax` en `get_partitions(..., scaler=...)`. El scaler se fittea **solo sobre `X_tr`** y se aplica con `transform` a validación y test (evita data-leakage). **`Y` nunca se escala**, para preservar la comparabilidad del MAE entre modelos y baselines.
- **Investigación** (`07_investigacion_ldp/`) — además aplica **diferenciación fraccional** (`d ≈ 0.4`) sobre las series de precio para obtener una representación estacionaria conservando memoria, según las ideas del primer taller del bloque.

## 4. Tarea de competición — modelos entrenados

Grid de ventanas: `v_in ∈ {5, 10, 30, 90}` × `v_out ∈ {1, 5, 30, 90}` → **16 combinaciones por arquitectura**. Cada combo vive en su propio notebook `{arch}_vin{V}_vout{V}.ipynb`, y un `_agregador_{arch}.ipynb` consolida los 16 parciales en CSV maestro y genera los heatmaps.

| Carpeta | Arquitectura | Idea |
|---|---|---|
| `02_baselines/` | Trivial (ceros), regresión lineal, Buy & Hold | Referencias para evaluar si un modelo aporta valor real. |
| `03_mlp/` | Capas densas | Aplana la ventana y predice los 23 retornos medios. |
| `04_recurrente/` | LSTM / GRU | Procesa la ventana como secuencia temporal. |
| `05_convolucional/` | Conv1D | Extrae patrones locales en el eje temporal. |
| `06_mixto/` | Conv + Recurrente | Combina extracción local y dependencia secuencial. |

Para cada combo se reportan:
- **MAE en train / val / test** (siempre los tres).
- **Número de parámetros** del modelo.
- **Curva de entrenamiento** con líneas de los 3 baselines superpuestas (verificación de convergencia).

Todos los artefactos se escriben en `08_results/`:

```
08_results/
├── tablas/parciales/         # 1 CSV por (arch, v_in, v_out)
├── tablas/{arch}_resultados.csv   # consolidado por arquitectura (16 filas)
├── runs/{arch}/{arch}_vinX_voutY/ # best.keras + best.json + history.csv
├── curvas/{arch}/            # PNG por combo, baselines superpuestos
├── matrices/                 # heatmaps 4×4 por split y arquitectura
└── baselines/                # diagnósticos de los baselines
```

La **matriz final 4×4 del mejor modelo en test** (resultado de la competición) se genera desde el agregador y aparece en la presentación.

## 5. Tarea de investigación

`07_investigacion_ldp/` repite las cuatro arquitecturas (MLP, RNN, CNN, mixto) aplicando el preprocesado del primer taller del bloque 3 (**diferenciación fraccional**, `d ≈ 0.4`). El foco está en `v_out = 90`, ya que el mejor modelo a 90 días es el que alimenta la cartera del apartado siguiente.

## 6. Cartera final — comparativa 2025

`11_Portfolio/04_full_pipeline.ipynb` implementa un pipeline unificado de inferencia y backtesting para 2025:

- **Cartera sin predicciones** — benchmarks equiponderados (BM1 fijo, BM2 con drift).
- **Cartera con predicciones** — usa el mejor modelo de la investigación a `v_out = 90`. Cada `v_out` días: predice el retorno esperado de los 23 activos, selecciona el **top-K** con esperanza positiva y rebalancea. Se prueban tres variantes:
  - **A** — equiponderada sobre el top-K.
  - **B** — equiponderada con *take-profit* sobre umbral.
  - **C** — peso proporcional al retorno esperado.

Salidas en cada subcarpeta `inv_{arch}_vin{V}_vout{V}/`: `summary_returns.csv`, `nav_series.csv`, curvas de NAV acumulado, scatter de predicción vs realización, y `metadata.json` con la configuración del modelo. La comparativa final muestra NAV de las tres estrategias frente a los dos benchmarks, y la correlación de Spearman entre predicción y realización por ventana.

## 7. Estructura del repositorio

```
.
├── 00_datos/                 # Lectura de datos (notebook proporcionado)
├── 01_src_compartido/        # dataset_utils.py, metrics_utils.py
├── 02_baselines/             # Trivial, lineal, buy & hold
├── 03_mlp/                   # 16 notebooks + agregador
├── 04_recurrente/            # 16 notebooks + agregador
├── 05_convolucional/         # 16 notebooks + agregador
├── 06_mixto/                 # 16 notebooks + agregador
├── 07_investigacion_ldp/     # Misma rejilla con diferenciación fraccional
├── 08_results/               # Tablas, curvas, heatmaps, checkpoints
├── 09_presentacion/          # PDF de presentación (5 min)
├── 10_instrucciones/         # Enunciado oficial (Taller_B3_T4.pdf)
└── 11_Portfolio/             # Pipeline de cartera 2025
```

## 8. Cómo ejecutar

Cada notebook comienza detectando el entorno y montando Drive sólo si corre en Colab; en local usa el directorio actual. El patrón está documentado en `10_instrucciones/CLAUDE.md`. Una vez fijado `BASE`, se importan las utilidades compartidas:

```python
from dataset_utils import load_data, create_dataset, get_partitions, TICKERS
from metrics_utils import plot_curva, BestRunTracker, resumen_vs_baselines
```

Flujo recomendado:
1. Abrir el notebook del combo deseado (p. ej. `04_recurrente/rnn_vin30_vout5.ipynb`).
2. *Restart & Run All*. El `BestRunTracker` persiste automáticamente el mejor modelo histórico y actualiza el CSV parcial.
3. Tras los 16 combos, lanzar `_agregador_{arch}.ipynb` para generar el CSV consolidado y los heatmaps.
4. Para la cartera, ejecutar `11_Portfolio/04_full_pipeline.ipynb` apuntando al modelo deseado.

## 9. Entregables

- **GitHub** (este repositorio) — 30% de la nota.
- **PDF de presentación** (5 min) en `09_presentacion/` — 70% de la nota.

Enunciado oficial: [10_instrucciones/Taller_B3_T4.pdf](10_instrucciones/Taller_B3_T4.pdf).
