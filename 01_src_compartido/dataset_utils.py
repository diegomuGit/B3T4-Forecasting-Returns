"""
dataset_utils.py
Taller B3-T4 — Funciones compartidas de carga y preprocesado de datos.

Reglas del enunciado (fiel a `00_datos/Lectura_datos_Taller_B3_T4.ipynb`):
- 23 activos del SP500 (los exactos del fichero de referencia).
- Datos de entrada: **log-retornos diarios** (`np.log(close).diff().dropna()`).
- Target Y: **promedio de los log-retornos** durante la ventana de salida.
- Partición train/test 90/10 **aleatoria** (`shuffle=True`), semilla 42.

Funciona en Colab y en local sin cambios. Importar con:
    from dataset_utils import load_data, create_dataset, get_partitions, TICKERS

──────────────────────────────────────────────────────────────────────────────
 EXTENSIÓN PARA INVESTIGACIÓN (López de Prado · `07_investigacion_ldp/`)
──────────────────────────────────────────────────────────────────────────────
`load_data` admite opcionalmente diferenciación fraccional (FFD, Fixed-Width
Frac Diff, López de Prado 2018) sobre `log(precio)` mediante el parámetro
`d_frac`. Los 64 notebooks de competición NO necesitan cambios: si `d_frac`
no se especifica (default `None`), se conserva el comportamiento original.

    d_frac=None  → modo competición (log-retornos si use_log_returns=True)
    d_frac=0.0   → log(precio) sin diferenciar
    d_frac=1.0   → log-retornos vía convolución (≈ caso de competición)
    d_frac∈(0,1) → frac-diff de log(precio): estacionaria con memoria larga

ATENCIÓN — Cuestiones que afectan al MAE en la investigación
------------------------------------------------------------
1) **Unidades del MAE.** Con `d_frac` activo, `create_dataset` deriva X e Y
   de la misma serie transformada, así que el MAE de los modelos de
   investigación está en *unidades de frac-diff*, no en log-retornos. Es
   internamente coherente pero **no directamente comparable** en valor
   absoluto con el MAE de los notebooks de competición. En el report,
   compárese siempre como *mejora relativa* contra un baseline trivial
   (predicción cero) calculado en la misma escala.

2) **Pérdida de muestras al inicio.** El método FFD descarta las primeras
   `L-1` filas del histórico (siendo `L` el número de pesos ω_k por encima
   de `fracdiff_threshold`). Para `d=0.4` y `threshold=1e-5`, `L` ronda los
   varios cientos / pocos miles; con ~16.200 días de histórico la pérdida
   típica es 1-10%. `load_data(..., verbose=True)` lo imprime al cargar.

3) **Elección del `d`.** Se decide UNA vez en
   `07_investigacion_ldp/00_eleccion_d_frac.ipynb` (ADF sobre el segmento
   de train cronológico, no sobre el histórico completo, para evitar
   peeking de test). El valor resultante se guarda como `D_FRAC_INV` aquí
   abajo y se importa desde los 16 notebooks de investigación:

       from dataset_utils import load_data, D_FRAC_INV
       data, df = load_data(d_frac=D_FRAC_INV)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── Tickers — 23 activos del SP500 ────────────────────────────────────────────
# Lista LITERAL de Lectura_datos_Taller_B3_T4.ipynb
TICKERS = [
    'AEP', 'BA', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED',
    'GD', 'GE', 'HON', 'HPQ', 'IBM', 'IP', 'JNJ', 'KO',
    'KR', 'MMM', 'MO', 'MRK', 'MSI', 'PG', 'XOM',
]
N_ASSETS = len(TICKERS)           # 23

# Semilla del notebook de referencia (`Lectura_datos_Taller_B3_T4.ipynb`, cell 8)
RANDOM_STATE_COMP = 42

# Semilla interna del split train/val (aleatorio, `shuffle=True`)
RANDOM_STATE_VAL = 123

# ── Investigación LdP: constantes de diferenciación fraccional ────────────────
# Valor PROVISIONAL del d hasta confirmar con `07_investigacion_ldp/00_eleccion_d_frac.ipynb`.
# En el taller previo (`Preprocesado_ML_Financiero.ipynb`) el d óptimo para BTC
# fue 0.4 (mínimo d con p-valor ADF < 0.05). Usamos 0.4 como punto de partida;
# el notebook de exploración refinará el valor sobre los 23 activos del taller.
D_FRAC_INV = 0.4
FRACDIFF_THRESHOLD_DEFAULT = 1e-5


# ── Diferenciación fraccional (FFD, López de Prado 2018) ──────────────────────

def calcular_pesos(d, threshold=FRACDIFF_THRESHOLD_DEFAULT, max_lag=20000):
    """
    Pesos ω_k de la expansión binomial generalizada (1-B)^d.

    Recurrencia:
        ω_0 = 1
        ω_k = -ω_{k-1} * (d - k + 1) / k        para k = 1, 2, 3, ...

    Se trunca al primer |ω_k| < `threshold` (FFD = Fixed-Width Frac Diff).
    `max_lag` actúa como safety cap.
    """
    pesos = [1.0]
    for k in range(1, max_lag):
        nuevo = -pesos[-1] * (d - k + 1) / k
        if abs(nuevo) < threshold:
            break
        pesos.append(nuevo)
    return np.array(pesos, dtype=np.float64)


def aplicar_fracdiff_multi(arr_2d, d, threshold=FRACDIFF_THRESHOLD_DEFAULT):
    """
    Frac-diff de orden `d` sobre cada columna de `arr_2d` (T, N).

    Implementación vectorizada por columna vía `np.convolve(..., mode='valid')`
    con los pesos de `calcular_pesos`. Equivalente exacto a la definición
    secuencial pero O(T·L·N) con constante baja.

    Devuelve
    --------
    out : np.ndarray (T - L + 1, N)
    L   : int  — número de pesos retenidos (filas iniciales descartadas: L-1)
    """
    pesos = calcular_pesos(d, threshold=threshold)
    L = len(pesos)
    T, N = arr_2d.shape
    if T < L:
        raise ValueError(
            f"Histórico insuficiente: T={T} < L={L} pesos para d={d}, "
            f"threshold={threshold}. Reduce el threshold o amplía el histórico."
        )
    out = np.empty((T - L + 1, N), dtype=arr_2d.dtype)
    for j in range(N):
        out[:, j] = np.convolve(arr_2d[:, j], pesos, mode='valid')
    return out, L


# ── load_data ─────────────────────────────────────────────────────────────────

def load_data(tickers=None, start='1945-01-01', end=None,
              use_log_returns=True,
              d_frac=None,
              fracdiff_threshold=FRACDIFF_THRESHOLD_DEFAULT,
              verbose=False):
    """
    Descarga precios de cierre ajustados desde yfinance y devuelve la
    transformación elegida.

    Modos (excluyentes; si se especifica `d_frac`, ignora `use_log_returns`):
    ------------------------------------------------------------------------
    `d_frac is None` (default — comportamiento competición):
        - `use_log_returns=True`  → log-retornos diarios `np.log(close).diff()`
        - `use_log_returns=False` → cierres ajustados sin transformar

    `d_frac` especificado — frac-diff (FFD) sobre `log(close)`:
        - `d_frac=0.0`   → `log(close)` (sin diferenciar)
        - `d_frac=1.0`   → log-retornos vía convolución (≈ caso competición)
        - `d_frac∈(0,1)` → frac-diff con memoria larga (uso típico LdP)

    Parámetros
    ----------
    tickers              : lista de tickers (default: los 23 del fichero).
    start                : fecha de inicio (default '1945-01-01').
    end                  : fecha de fin (default None → hasta hoy).
    use_log_returns      : ver tabla de modos arriba; default True.
    d_frac               : orden de frac-diff. None → modo competición.
    fracdiff_threshold   : truncación de los pesos ω_k (FFD). Default 1e-5.
    verbose              : prints de progreso; default False.

    Devuelve
    --------
    data : np.ndarray (T, n_assets) float32
    df   : pd.DataFrame con índice de fechas y misma información

    Notas
    -----
    Con `d_frac` activo:
    - Se descartan las primeras `L-1` filas del histórico (truncación FFD).
    - El MAE de los modelos quedará en *unidades de frac-diff*, no en
      log-retornos. Ver el encabezado del módulo para más detalle.
    """
    if tickers is None:
        tickers = TICKERS

    if verbose:
        print(f'> Descargando datos desde {start} hasta {end if end else "hoy"}...')
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=verbose)['Close']

    raw = raw.dropna(axis=1)
    raw = raw[[t for t in tickers if t in raw.columns]]

    if d_frac is not None:
        log_price = np.log(raw).dropna()
        fd, L = aplicar_fracdiff_multi(
            log_price.values.astype(np.float64),
            d=d_frac,
            threshold=fracdiff_threshold,
        )
        df = pd.DataFrame(
            fd,
            index=log_price.index[L - 1:],
            columns=log_price.columns,
        )
        if verbose:
            n_lost = L - 1
            pct_lost = 100.0 * n_lost / max(len(log_price), 1)
            print(f'> Frac-diff aplicada — d={d_frac}, L={L} pesos, '
                  f'{n_lost} filas iniciales descartadas '
                  f'(~{pct_lost:.2f}% del histórico).')
            print(f'> Shape resultante: {df.shape}')
    elif use_log_returns:
        df = np.log(raw).diff().dropna()
        if verbose:
            print(f'> Log-retornos calculados — shape {df.shape}')
    else:
        df = raw.copy()
        if verbose:
            print(f'> Cierres ajustados — shape {df.shape}')

    data = df.values.astype(np.float32)
    return data, df


# ── create_dataset ────────────────────────────────────────────────────────────

def create_dataset(data, v_in, v_out, verbose=False):
    """
    Construye el dataset de ventanas deslizantes para forecasting.

    X[i] = data[i : i + v_in]                       (v_in, Ch)
    Y[i] = data[i + v_in : i + v_in + v_out].mean() (Ch,)

    Parámetros
    ----------
    data    : np.ndarray (T, Ch)
    v_in    : int — días de ventana de entrada
    v_out   : int — días de ventana de salida
    verbose : si True imprime las shapes resultantes; default False.

    Devuelve
    --------
    X : np.ndarray (N, v_in, Ch)
    Y : np.ndarray (N, Ch)       — promedio temporal de la ventana de salida
    """
    X, Y = [], []
    T = len(data)

    for i in range(T - v_in - v_out + 1):
        x_window = data[i : i + v_in]
        y_window = data[i + v_in : i + v_in + v_out]
        X.append(x_window)
        Y.append(y_window.mean(axis=0))

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    if verbose:
        print(f'> Dataset creado — X: {X.shape}, Y: {Y.shape}')
    return X, Y


# ── get_partitions ────────────────────────────────────────────────────────────

def get_partitions(X, Y, test_size=0.1, val_size=0.1,
                   random_state_comp=RANDOM_STATE_COMP,
                   random_state_val=RANDOM_STATE_VAL,
                   scaler=None,
                   return_scaler=False,
                   verbose=False):
    """
    Partición oficial 90/10 de la competición — **aleatoria** (`shuffle=True`),
    fiel al notebook de referencia (cell 8: `train_test_split(..., shuffle=True, random_state=42)`).
    Añade además un split train/val interno (también aleatorio, `shuffle=True`,
    `random_state=123`) para monitorizar el entrenamiento con
    `validation_data` y EarlyStopping.

    El test queda intacto y reproducible para todos los modelos.

    Parámetros
    ----------
    scaler : None | 'standard' | 'minmax'
        Si se indica, fittea el scaler **solo sobre `X_tr`** (aplanado a
        (N*v_in, Ch)) y aplica `transform` a X_tr / X_val / X_test.
        Nunca fittea sobre val/test (eso sería data-leakage).
        Y queda sin escalar para preservar la comparabilidad del MAE con
        baselines y otros modelos.
    return_scaler : bool
        Si True devuelve un 7º valor con el scaler ajustado (o None).
        Default False → retro-compatible con la firma original de 6 valores.
    verbose : si True imprime los tamaños de cada partición; default False.

    Devuelve
    --------
    X_tr, X_val, X_test, Y_tr, Y_val, Y_test                (return_scaler=False)
    X_tr, X_val, X_test, Y_tr, Y_val, Y_test, scaler_obj    (return_scaler=True)
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state_comp,
        shuffle=True,
    )

    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_train, Y_train,
        test_size=val_size,
        random_state=random_state_val,
        shuffle=True,
    )

    # ── Escalado opcional (fit SOLO con X_tr) ─────────────────────────────────
    sc = None
    if scaler is not None:
        if scaler == 'standard':
            sc = StandardScaler()
        elif scaler == 'minmax':
            sc = MinMaxScaler()
        else:
            raise ValueError(f"scaler debe ser None, 'standard' o 'minmax'; recibido: {scaler!r}")

        n_ch = X_tr.shape[-1]
        sc.fit(X_tr.reshape(-1, n_ch))

        def _apply(arr):
            s = arr.shape
            return sc.transform(arr.reshape(-1, n_ch)).reshape(s).astype(np.float32)

        X_tr  = _apply(X_tr)
        X_val = _apply(X_val)
        X_test = _apply(X_test)

        if verbose:
            print(f'> Scaler: {scaler} fitted on X_tr only — applied to train/val/test')

    if verbose:
        print(f'> Particiones — Train: {X_tr.shape[0]:>6}  '
              f'Val: {X_val.shape[0]:>6}  Test: {X_test.shape[0]:>6}')

    if return_scaler:
        return X_tr, X_val, X_test, Y_tr, Y_val, Y_test, sc
    return X_tr, X_val, X_test, Y_tr, Y_val, Y_test


# ── get_partitions_temporal (investigación LdP) ───────────────────────────────

def get_partitions_temporal(X, Y, v_in, v_out,
                            test_frac=0.10, val_frac=0.10,
                            embargo=None,
                            scaler=None,
                            return_scaler=False,
                            random_state=RANDOM_STATE_VAL,
                            verbose=False):
    """
    Partición **temporal** train / val / test con embargo (purged split, LdP).

    Pensada para los 16 notebooks de investigación. NO sustituye a
    `get_partitions` (que mantiene la partición aleatoria oficial de la
    competición con `random_state=42`).

    Distribución cronológica
    ------------------------
        | TRAIN       | embargo | VAL  | embargo | TEST |
        ^ inicio                                         ^ fin del histórico

    - `test_frac` y `val_frac` son fracciones del total `N` de muestras.
    - El `embargo` (default `v_in + v_out`) descarta las muestras de frontera
      cuyas ventanas (entrada Y/o salida) se solaparían con la partición
      adyacente. Es el equivalente del Purged K-Fold con embargo (López de
      Prado, *Advances in Financial Machine Learning*, cap. 7).

    Justificación del embargo `v_in + v_out`
    ----------------------------------------
    Cada muestra `i` está "activa" en el rango temporal `[i, i+v_in+v_out-1]`
    (X usa `[i, i+v_in)` e Y usa `[i+v_in, i+v_in+v_out)`). Para que ninguna
    muestra de train solape con ninguna muestra posterior basta con dejar un
    gap de `v_in + v_out` posiciones entre el último índice de train y el
    primero de la partición siguiente. Mismo razonamiento entre val y test.

    Parámetros
    ----------
    X, Y         : np.ndarray salida de `create_dataset`. Se asume orden
                    cronológico (el constructor `create_dataset` lo respeta).
    v_in, v_out  : ventanas de entrada y salida usadas en `create_dataset`.
    test_frac    : fracción del total para test (default 0.10).
    val_frac     : fracción del total para val  (default 0.10).
    embargo      : tamaño del embargo en muestras. Si None → `v_in + v_out`.
    scaler       : None | 'standard' | 'minmax'. Si se indica, se ajusta
                    SOLO sobre X_tr y se aplica a X_tr / X_val / X_test.
    return_scaler: si True, devuelve el scaler como 7º valor.
    random_state : semilla. NO afecta al split (es determinista por construcción);
                    se acepta por coherencia con la firma de `get_partitions`
                    (legacy) y para futuras extensiones que sí usen aleatoriedad
                    (p.ej. sample weighting). Default `RANDOM_STATE_VAL` (=123).
    verbose      : prints de tamaños y rangos.

    Devuelve
    --------
    X_tr, X_val, X_test, Y_tr, Y_val, Y_test                (return_scaler=False)
    X_tr, X_val, X_test, Y_tr, Y_val, Y_test, scaler_obj    (return_scaler=True)
    """
    if embargo is None:
        embargo = v_in + v_out

    # Aceptamos `random_state` por coherencia con `get_partitions` pero no
    # se utiliza: el split es determinista. Lo evitamos como variable no
    # leída en algunos linters con un `del` explícito.
    del random_state

    N = X.shape[0]
    n_test  = int(round(test_frac * N))
    n_val   = int(round(val_frac  * N))
    n_train = N - n_test - n_val - 2 * embargo

    if n_train <= 0:
        raise ValueError(
            f"Partición temporal vacía: N={N}, test={n_test}, val={n_val}, "
            f"2·embargo={2*embargo} → train={n_train}. "
            f"Reduce val_frac/test_frac o el embargo."
        )

    i0_tr,  i1_tr  = 0, n_train
    i0_val, i1_val = i1_tr  + embargo, i1_tr  + embargo + n_val
    i0_te,  i1_te  = i1_val + embargo, i1_val + embargo + n_test

    X_tr,   Y_tr   = X[i0_tr : i1_tr],    Y[i0_tr : i1_tr]
    X_val,  Y_val  = X[i0_val: i1_val],   Y[i0_val: i1_val]
    X_test, Y_test = X[i0_te : i1_te],    Y[i0_te : i1_te]

    # ── Escalado opcional (fit SOLO con X_tr) ─────────────────────────────────
    sc = None
    if scaler is not None:
        if scaler == 'standard':
            sc = StandardScaler()
        elif scaler == 'minmax':
            sc = MinMaxScaler()
        else:
            raise ValueError(f"scaler debe ser None, 'standard' o 'minmax'; recibido: {scaler!r}")

        n_ch = X_tr.shape[-1]
        sc.fit(X_tr.reshape(-1, n_ch))

        def _apply(arr):
            s = arr.shape
            return sc.transform(arr.reshape(-1, n_ch)).reshape(s).astype(np.float32)

        X_tr   = _apply(X_tr)
        X_val  = _apply(X_val)
        X_test = _apply(X_test)

        if verbose:
            print(f'> Scaler: {scaler} fitted on X_tr only — applied to train/val/test')

    if verbose:
        print(f'> Split temporal con embargo  '
              f'(v_in={v_in}, v_out={v_out}, embargo={embargo})')
        print(f'  Train   [{i0_tr :>6}, {i1_tr :>6})  ->  {X_tr.shape[0] :>6} muestras')
        print(f'  Val     [{i0_val:>6}, {i1_val:>6})  ->  {X_val.shape[0]:>6} muestras  '
              f'(gap previo: {embargo})')
        print(f'  Test    [{i0_te :>6}, {i1_te :>6})  ->  {X_test.shape[0]:>6} muestras  '
              f'(gap previo: {embargo})')
        descartadas = N - X_tr.shape[0] - X_val.shape[0] - X_test.shape[0]
        print(f'  Descartadas por embargo: {descartadas} muestras  '
              f'({100*descartadas/N:.2f}% del total)')

    if return_scaler:
        return X_tr, X_val, X_test, Y_tr, Y_val, Y_test, sc
    return X_tr, X_val, X_test, Y_tr, Y_val, Y_test
