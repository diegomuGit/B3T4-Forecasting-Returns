# Taller B3-T4 — Redes Neuronales para Forecasting

## Project context

Academic project (MIAX program). Deadline: **21 May, 18:00**.
Three-person team. Deliverables: GitHub repo + PDF presentation (5 min strict).
Grading: 30% GitHub, 70% presentation.

## Task constraints — READ BEFORE CODING

- **Competition seed is FIXED**: always use `random_state=42` for train/test split (matches the reference notebook `00_datos/Lectura_datos_Taller_B3_T4.ipynb`). Never change it.
- **Split is chronological**: `train_test_split(..., shuffle=False)`. Test = last 10% in time order.
- **Minimum 64 models**: 4 architectures (MLP, LSTM/GRU, Conv1D, mixed) × 16 window combos.
- **Window grid**: `v_in` ∈ {5, 10, 30, 90} days, `v_out` ∈ {1, 5, 30, 90} days.
- **Always report MAE on train, val, AND test**. Never report only one or two.
- **Input X**: daily **log-returns** (`np.log(close).diff().dropna()`), shape `(N, v_in, 23)`. Do not flatten unless the model requires it (e.g. MLP).
- **Target Y**: mean of **log-returns** over the `v_out` future days, shape `(N, 23)`.
- **Escalado opcional**: si se usa `scaler='standard'|'minmax'` en `get_partitions()`, el scaler **se fittea solo sobre `X_tr`** y se aplica con `transform` a val/test. Nunca fittear sobre val/test ni sobre el array completo (data-leakage). **Y queda siempre sin escalar** para preservar la comparabilidad del MAE con baselines y entre modelos.

## Environment detection — CRITICAL PATTERN

Every notebook MUST start with this cell. Never hardcode Colab or local paths:

```python
import os, sys

def detect_env():
    try:
        import google.colab
        return 'colab'
    except ImportError:
        return 'local'

ENV = detect_env()

if ENV == 'colab':
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/Taller4_DL_MIAX'
else:
    BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))

SRC = os.path.join(BASE, '01_src_compartido')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
```

Then import shared functions — same line works in both environments:

```python
from dataset_utils import load_data, create_dataset, get_partitions, TICKERS
import metrics_utils
metrics_utils.BASE_DRIVE = BASE   # inject path before calling any save function
from metrics_utils import (
    plot_curva,                   # admite mostrar_baselines=True (default)
    BestRunTracker,               # callback: persiste best/last + parcial automáticamente
    resumen_vs_baselines,         # tabla MAE modelo vs 3 baselines del combo
    cargar_best_modelo,           # recupera (model, info) de la mejor histórica
    # consolidar_resultados, plot_heatmap  ← solo en el agregador
)
```

**NEVER use** `from google.colab import drive` outside the `if ENV == 'colab'` block.
**NEVER use** `%run` to load shared code — use `import` from `01_src_compartido/*.py` instead.

## Shared utilities location

```
01_src_compartido/
├── dataset_utils.py   # load_data, create_dataset, TICKERS, RANDOM_STATE_COMP
                       # get_partitions(..., scaler=None, return_scaler=False)
                       #   scaler ∈ {None, 'standard', 'minmax'} — fit SOLO sobre X_tr
└── metrics_utils.py   # calc_mae, calc_mae_all,
                       # plot_curva (con mostrar_baselines=True por defecto),
                       # plot_heatmap, plot_comparativa_vout, plot_matriz_final,
                       # guardar_resultado_parcial   (1 fila por notebook)
                       # consolidar_resultados       (16 parciales → CSV maestro)
                       # BestRunTracker              (callback: best/last + history + epoch_history)
                       #   → auto_plot=False         pasar True para guardar curva PNG automáticamente si mejora
                       # cargar_best_modelo          (recupera best.keras + best.json)
                       # cargar_baselines_combo      (lookup MAE baselines (v_in,v_out))
                       # resumen_vs_baselines        (tabla comparativa modelo vs baselines)
                       # guardar_resultados          (legacy: aún usado por baselines)
```

`metrics_utils.BASE_DRIVE` must be set before any function that saves files is called.

## Results output structure

All saved files go under `BASE/08_results/`:

```
08_results/
├── tablas/
│   ├── parciales/                    # 1 CSV per (arch, v_in, v_out) → 64 al final
│   │   └── {modelo}_vin{V}_vout{V}.csv   (1 fila — refleja la MEJOR ejecución histórica del combo)
│   ├── baseline_*.csv                # CSVs de los 3 baselines (buy_and_hold, trivial, lineal)
│   └── {modelo}_resultados.csv       # consolidado por arquitectura (escribe el agregador)
├── runs/                             # tracking best/last por combo (lo escribe BestRunTracker)
│   └── {modelo}/{modelo}_vin{V}_vout{V}/
│       ├── best.keras                # arch+pesos+optimizer de la mejor ejecución histórica
│       ├── best.json                 # hparams + MAEs (train/val/test) + epoch + duration_seconds + timestamp
│       ├── last.json                 # auditoría de la última ejecución (siempre; misma estructura que best.json)
│       ├── history.csv               # append-only: 1 fila por ejecución (incluye overfit_ratio y duration_seconds)
│       └── epoch_history.csv         # append-only: 1 fila por época (timestamp, epoch, loss, val_loss)
├── matrices/                         # heatmaps 4x4 (PNG) — los genera el agregador
├── curvas/
│   └── {modelo}/                     # subcarpeta por arquitectura (16 PNGs cada una)
│       └── {modelo}_vin{V}_vout{V}.png   (curva con MAE de los 3 baselines superpuestos)
├── baselines/                        # diagnósticos de baselines (coef, overfitting, etc.)
│   └── lineal/                       # PNGs específicos del baseline lineal
└── graficas_combinaciones/           # comparativas por v_out (notebook global de presentación)
```

> Nota: `runs/` se crea automáticamente la primera vez que un notebook entrena
> con `BestRunTracker`. Si el repo aún no tiene la carpeta, es porque ningún
> combo se ha ejecutado todavía bajo este flujo — no es un error.

> ⚠️ **Keras Tuner está descartado**. Si aparecen `08_results/best_hps/` o
> `08_results/tuner/` en el repo, son restos de un experimento anterior y
> deben borrarse. Los hiperparámetros se fijan a mano en cada notebook.

**El CSV parcial se actualiza desde `best.json`**, no desde la última ejecución: si una nueva
run no mejora la `mae_val` histórica del combo, ni el parcial ni el `best.keras` cambian.

Use `Path(BASE) / '08_results' / ...` for paths. Call `ruta.parent.mkdir(parents=True, exist_ok=True)` before saving.

## Standard notebook structure (1 notebook = 1 combo)

Cada combinación `(v_in, v_out)` vive en su propio notebook
`{arch}_vin{V}_vout{V}.ipynb` dentro de la carpeta de su arquitectura
(`03_mlp/`, `04_recurrente/`, `05_convolucional/`, `06_mixto/`).
Total: **64 notebooks de modelos + 4 agregadores** (`_agregador_{modelo}.ipynb`).

Patrón de cada notebook individual (sin bucle, basado en `BestRunTracker`):

```python
# Constantes locales (cambiar V_IN, V_OUT por combo):
MODELO = 'mlp'      # o 'lstm', 'conv1d', 'mixto'
V_IN, V_OUT = 5, 1
EPOCHS, BATCH_SIZE, PATIENCE = 100, 100, 15

# Carga, dataset, partición:
data, df = load_data()
X, Y     = create_dataset(data, V_IN, V_OUT)
X_tr, X_val, X_test, Y_tr, Y_val, Y_test = get_partitions(X, Y)
# Con escalado (fit solo sobre X_tr; Y no se escala):
# X_tr, X_val, X_test, Y_tr, Y_val, Y_test, sc = get_partitions(
#     X, Y, scaler='standard', return_scaler=True
# )

# Modelo + entrenamiento (BestRunTracker persiste best/last + parcial):
model   = build_<arch>(V_IN, ...)
tracker = BestRunTracker(
    MODELO, V_IN, V_OUT,
    datasets=(X_tr, Y_tr, X_val, Y_val, X_test, Y_test),
    hparams_extra={'batch_size': BATCH_SIZE, 'patience': PATIENCE, 'epochs_max': EPOCHS},
    # auto_plot=True  ← descomenta para guardar curva automáticamente cuando mejora
)
history = model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[tracker], verbose=0)

# Visualización (curva con baselines superpuestos + tabla comparativa):
# save=tracker.improved → solo escribe el PNG si esta run batió el mae_val histórico
plot_curva(history, MODELO, V_IN, V_OUT, mostrar_baselines=True, save=tracker.improved)
resumen_vs_baselines(tracker.maes_last, V_IN, V_OUT, nombre_modelo=MODELO)
```

Para recuperar la arquitectura + pesos + hparams de la mejor ejecución histórica de un combo:

```python
model_best, info = cargar_best_modelo('mlp', V_IN=5, V_OUT=1)
# info['hparams'] contiene arch, optimizer config y hparams_extra (batch, patience, ...)
```

Patrón del agregador (`_agregador_{modelo}.ipynb`, se ejecuta una vez los 16 parciales están listos):

```python
from metrics_utils import consolidar_resultados, plot_heatmap

df = consolidar_resultados('mlp')   # lee 16 parciales → mlp_resultados.csv
for split in ['train', 'val', 'test']:
    plot_heatmap(df, 'mlp', split=split, save=True)
```

## Workflow

1. **Cada combinación = un notebook independiente** (`{arch}_vin{V}_vout{V}.ipynb`).
   Desarrollar y entrenar en **Google Colab** (GPU). El notebook escribe su CSV
   parcial y su PNG de curva inmediatamente al terminar.
2. Tras ejecutar los 16 notebooks de una arquitectura, correr
   `_agregador_{arch}.ipynb` para consolidar el CSV maestro y generar los heatmaps.
3. Restart & Run All por notebook antes de commit. Solo commitear notebooks limpios.
4. **GitHub es el delivery final**: 64 notebooks de modelos + 4 agregadores. No drafts.

## Verification checklist before committing to GitHub

- [ ] Cada notebook individual corre Restart & Run All sin errores.
- [ ] `08_results/tablas/parciales/` contiene 64 CSVs (4 arquitecturas × 16 combos), cada uno con 1 fila.
- [ ] `08_results/runs/{modelo}/{modelo}_vinX_voutY/` existe con `best.keras` + `best.json` + `last.json` + `history.csv` + `epoch_history.csv` para los 64 combos.
- [ ] `08_results/tablas/{modelo}_resultados.csv` tiene 16 filas tras correr el agregador.
- [ ] `08_results/curvas/{modelo}/` tiene 16 PNGs por arquitectura, con líneas de baselines visibles.
- [ ] `08_results/matrices/` tiene heatmaps de los 3 splits para cada arquitectura.
- [ ] Curvas de entrenamiento muestran convergencia (val_loss aplana o EarlyStopping dispara).
- [ ] No se commitean CSVs de yfinance (los datos siempre se descargan por código).
- [ ] **No** existen `08_results/best_hps/` ni `08_results/tuner/` (Keras Tuner descartado).
- [ ] Diagnósticos de baselines viven en `08_results/baselines/{baseline}/`, **no** en `graficas_combinaciones/` (esa carpeta es solo para las comparativas modelo-vs-modelo por v_out).
