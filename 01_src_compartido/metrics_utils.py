"""
metrics_utils.py
Taller B3-T4 — Funciones compartidas de métricas y visualización.

Funciona en Colab y en local sin cambios. Importar con:
    from metrics_utils import calc_mae, calc_mae_all, plot_curva, plot_heatmap, ...

BASE_DRIVE se inyecta desde cada notebook antes de llamar a funciones que guardan ficheros:
    import metrics_utils
    metrics_utils.BASE_DRIVE = BASE   # BASE viene de env_setup
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Nombres de los CSV de baselines bajo 08_results/tablas/.
# Schema esperado: modelo, v_in, v_out, n_params, mae_train, mae_val, mae_test.
BASELINE_FILES = {
    'buy_and_hold': 'baseline_buy_and_hold_resultados.csv',
    'trivial':      'baseline_trivial_resultados.csv',
    'lineal':       'baseline_lineal_resultados.csv',
}
# Variantes en escala de investigación (frac-diff + split temporal con embargo).
# Se consultan automáticamente desde los notebooks `inv_*` (auto-detect por
# `nombre_modelo.startswith('inv_')` en `plot_curva` / `resumen_vs_baselines`).
BASELINE_FILES_INV = {
    'buy_and_hold': 'baseline_buy_and_hold_inv_resultados.csv',
    'trivial':      'baseline_trivial_inv_resultados.csv',
    'lineal':       'baseline_lineal_inv_resultados.csv',
}
BASELINE_COLORS = {
    'buy_and_hold': '#1f77b4',
    'trivial':      '#2ca02c',
    'lineal':       '#9467bd',
}

# BASE_DRIVE se sobreescribe desde cada notebook via: metrics_utils.BASE_DRIVE = BASE
# Por defecto apunta a la carpeta de resultados relativa al CWD (útil en local)
BASE_DRIVE = str(Path('.').resolve().parent)

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10
})


# ── calc_mae ──────────────────────────────────────────────────────────────────

def calc_mae(model, X, Y, verbose=False):
    """MAE de un modelo Keras sobre un conjunto de datos."""
    Y_pred = model.predict(X, verbose=0)
    mae = float(np.mean(np.abs(Y_pred - Y)))
    if verbose:
        print(f'MAE: {mae:.6f}')
    return mae


def calc_mae_all(model, X_tr, Y_tr, X_val, Y_val, X_test, Y_test):
    """MAE en train, val y test de una vez. Devuelve dict."""
    return {
        'train': calc_mae(model, X_tr,   Y_tr),
        'val'  : calc_mae(model, X_val,  Y_val),
        'test' : calc_mae(model, X_test, Y_test)
    }


# ── plot_curva ────────────────────────────────────────────────────────────────

def plot_curva(history, nombre_modelo, v_in, v_out, save=True,
               mostrar_baselines=True, verbose=False):
    """
    Curva de entrenamiento train vs val.
    Si mostrar_baselines=True, añade líneas horizontales con el mae_val de los
    3 baselines (buy_and_hold / trivial / lineal) para el mismo (v_in, v_out).
    Si los CSV de baselines no existen, degrada silenciosamente.
    Guarda PNG en BASE_DRIVE/08_results/curvas/ si save=True.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(history.history['loss'],     label='Train loss', linewidth=1.5)
    ax.plot(history.history['val_loss'], label='Val loss',   linewidth=1.5, linestyle='--')

    best_epoch = int(np.argmin(history.history['val_loss']))
    best_val   = history.history['val_loss'][best_epoch]
    ax.axvline(best_epoch, color='red', linewidth=0.8, linestyle=':',
               label=f'Best epoch ({best_epoch})')

    if mostrar_baselines:
        scope = 'inv' if str(nombre_modelo).startswith('inv_') else 'comp'
        df_b = cargar_baselines_combo(v_in, v_out, scope=scope)
        for _, row in df_b.iterrows():
            key   = row['baseline']
            color = BASELINE_COLORS.get(key, 'gray')
            ax.axhline(row['mae_val'], color=color, linewidth=0.9, linestyle='-.',
                       alpha=0.7,
                       label=f'{key} (val={row["mae_val"]:.5f}, test={row["mae_test"]:.5f})')

    ax.set_title(f'{nombre_modelo} — V_in={v_in}, V_out={v_out} | Best val={best_val:.5f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE loss')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        nombre = nombre_modelo.lower().replace(' ', '_')
        ruta = Path(BASE_DRIVE) / '08_results' / 'curvas' / nombre / f'{nombre}_vin{v_in}_vout{v_out}.png'
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
        if verbose:
            print(f'> Curva guardada: {ruta}')

    return fig


# ── plot_heatmap ──────────────────────────────────────────────────────────────

def plot_heatmap(df_resultados, nombre_modelo, split='test', save=True, verbose=False):
    """
    Heatmap 4×4 de MAE para un modelo.
    df_resultados debe tener columnas: modelo, v_in, v_out, mae_train, mae_val, mae_test
    Silencioso por defecto; usar verbose=True para ver la ruta del PNG.
    """
    df = df_resultados[df_resultados['modelo'] == nombre_modelo].copy()
    pivot = df.pivot(index='v_in', columns='v_out', values=f'mae_{split}')

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot,
        annot=True, fmt='.5f',
        cmap='YlOrRd',
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': f'MAE {split}'}
    )
    ax.set_title(f'{nombre_modelo} — MAE {split}')
    ax.set_xlabel('V_out (días de salida)')
    ax.set_ylabel('V_in (días de entrada)')
    fig.tight_layout()

    if save:
        nombre = nombre_modelo.lower().replace(' ', '_')
        ruta = Path(BASE_DRIVE) / '08_results' / 'matrices' / f'heatmap_{nombre}_{split}.png'
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
        if verbose:
            print(f'> Heatmap guardado: {ruta}')

    return fig


# ── plot_comparativa_vout ─────────────────────────────────────────────────────

def plot_comparativa_vout(df_resultados, v_out, split='test', save=True, verbose=False):
    """
    Gráfica de barras comparando todos los modelos para un V_out dado.
    Genera la gráfica resumen por V_out que pide el enunciado.
    Silencioso por defecto; usar verbose=True para ver la ruta del PNG.
    """
    df      = df_resultados[df_resultados['v_out'] == v_out].copy()
    modelos = df['modelo'].unique()
    v_ins   = sorted(df['v_in'].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(v_ins))
    width = 0.8 / len(modelos)

    for i, modelo in enumerate(modelos):
        df_m    = df[df['modelo'] == modelo].set_index('v_in')
        valores = [df_m.loc[v, f'mae_{split}'] if v in df_m.index else np.nan for v in v_ins]
        ax.bar(x + i * width, valores, width=width, label=modelo)

    ax.set_xticks(x + width * (len(modelos) - 1) / 2)
    ax.set_xticklabels([f'V_in={v}' for v in v_ins])
    ax.set_title(f'Comparativa modelos — V_out={v_out} | MAE {split}')
    ax.set_ylabel(f'MAE {split}')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save:
        ruta = Path(BASE_DRIVE) / '08_results' / 'graficas_combinaciones' / f'comparativa_vout{v_out}_{split}.png'
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
        if verbose:
            print(f'> Comparativa guardada: {ruta}')

    return fig


# ── guardar_resultados ────────────────────────────────────────────────────────

def guardar_resultados(lista_resultados, nombre_modelo, verbose=False):
    """
    Legacy: convierte la lista de dicts del bucle 4×4 en DataFrame y guarda CSV.
    Aún en uso por los baselines (02_baselines/). Para el nuevo flujo
    "1 notebook = 1 combo" usar guardar_resultado_parcial + consolidar_resultados.
    Columnas estándar: modelo, v_in, v_out, n_params, mae_train, mae_val, mae_test
    """
    df   = pd.DataFrame(lista_resultados)
    cols = ['modelo', 'v_in', 'v_out', 'n_params', 'mae_train', 'mae_val', 'mae_test']
    df   = df[cols]

    nombre = nombre_modelo.lower().replace(' ', '_')
    ruta   = Path(BASE_DRIVE) / '08_results' / 'tablas' / f'{nombre}_resultados.csv'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)

    if verbose:
        print(f'> Resultados guardados: {ruta}')
        print(df.to_string(index=False))
    return df


# ── guardar_resultado_parcial ─────────────────────────────────────────────────

def guardar_resultado_parcial(resultado, nombre_modelo, v_in, v_out, verbose=False):
    """
    Guarda el resultado de UNA sola combinación (1 fila) en
    BASE_DRIVE/08_results/tablas/parciales/{modelo}_vin{V}_vout{V}.csv

    Pensado para el flujo "1 notebook = 1 combo": cada notebook escribe su
    propio CSV. Idempotente (sobrescribe). Después, consolidar_resultados()
    concatena los 16 parciales en el CSV maestro.

    resultado: dict con keys modelo, v_in, v_out, n_params, mae_train, mae_val, mae_test.
    """
    cols = ['modelo', 'v_in', 'v_out', 'n_params', 'mae_train', 'mae_val', 'mae_test']
    df   = pd.DataFrame([resultado])[cols]

    nombre = nombre_modelo.lower().replace(' ', '_')
    ruta   = Path(BASE_DRIVE) / '08_results' / 'tablas' / 'parciales' / f'{nombre}_vin{v_in}_vout{v_out}.csv'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)

    if verbose:
        print(f'\n> Resultado parcial guardado: {ruta}')
    return ruta


# ── consolidar_resultados ─────────────────────────────────────────────────────

def consolidar_resultados(nombre_modelo, verbose=False):
    """
    Lee todos los parciales {modelo}_vin*_vout*.csv en
    BASE_DRIVE/08_results/tablas/parciales/, los concatena ordenados por
    (v_in, v_out), y guarda el CSV maestro en
    BASE_DRIVE/08_results/tablas/{modelo}_resultados.csv.

    Si hay menos de 16 parciales, avisa con un warning pero no falla
    (permite ejecutar el agregador parcialmente durante el desarrollo).
    Devuelve el DataFrame consolidado.
    """
    nombre   = nombre_modelo.lower().replace(' ', '_')
    parciales = Path(BASE_DRIVE) / '08_results' / 'tablas' / 'parciales'
    archivos  = sorted(parciales.glob(f'{nombre}_vin*_vout*.csv'))

    if len(archivos) == 0:
        raise FileNotFoundError(
            f'No se encontraron parciales para "{nombre_modelo}" en {parciales}. '
            f'Ejecuta antes los notebooks individuales {nombre}_vin*_vout*.ipynb.'
        )
    if len(archivos) < 16:
        print(f'> WARNING: solo {len(archivos)}/16 parciales encontrados para "{nombre_modelo}".')

    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    df = df.sort_values(['v_in', 'v_out']).reset_index(drop=True)

    ruta = Path(BASE_DRIVE) / '08_results' / 'tablas' / f'{nombre}_resultados.csv'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)

    if verbose:
        print(f'> Consolidado guardado: {ruta} ({len(df)} filas)')
        print(df.to_string(index=False))
    return df


# ── plot_matriz_final ─────────────────────────────────────────────────────────

def plot_matriz_final(df_todos, save=True, verbose=False):
    """
    Matriz final 4×4 con el mejor MAE test por combinación de ventanas.
    Cada celda muestra el MAE y el nombre del modelo ganador.
    Silencioso por defecto; usar verbose=True para ver la ruta del PNG.
    """
    idx      = df_todos.groupby(['v_in', 'v_out'])['mae_test'].idxmin()
    df_mejor = df_todos.loc[idx].reset_index(drop=True)

    pivot_mae    = df_mejor.pivot(index='v_in', columns='v_out', values='mae_test')
    pivot_modelo = df_mejor.pivot(index='v_in', columns='v_out', values='modelo')
    annot        = pivot_mae.round(5).astype(str) + '\n(' + pivot_modelo + ')'

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot_mae,
        annot=annot, fmt='',
        cmap='YlGn_r',
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'MAE test (menor = mejor)'}
    )
    ax.set_title('Matriz final — Mejor MAE test por combinación\n(modelo ganador en cada celda)')
    ax.set_xlabel('V_out (días de salida)')
    ax.set_ylabel('V_in (días de entrada)')
    fig.tight_layout()

    if save:
        ruta = Path(BASE_DRIVE) / '08_results' / 'matrices' / 'matriz_final_competicion.png'
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
        if verbose:
            print(f'> Matriz final guardada: {ruta}')

    return fig, df_mejor


# ── runs/ best/last tracking ──────────────────────────────────────────────────

def _run_dir(nombre_modelo, v_in, v_out):
    """Carpeta de ejecuciones para un combo: 08_results/runs/{modelo}/{modelo}_vinX_voutY/."""
    nombre = nombre_modelo.lower().replace(' ', '_')
    return (Path(BASE_DRIVE) / '08_results' / 'runs' / nombre /
            f'{nombre}_vin{v_in}_vout{v_out}')


def cargar_best_modelo(nombre_modelo, v_in, v_out):
    """
    Recupera la mejor ejecución histórica para un combo.
    Devuelve (model, info_dict). info_dict es el contenido de best.json
    (incluye hparams, mae_train/val/test, best_epoch, timestamp).
    """
    from tensorflow import keras  # import lazy: solo si se usa

    rdir = _run_dir(nombre_modelo, v_in, v_out)
    keras_path = rdir / 'best.keras'
    json_path  = rdir / 'best.json'
    if not keras_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f'No hay best.* en {rdir}. ¿Has entrenado este combo con BestRunTracker?'
        )
    model = keras.models.load_model(keras_path)
    info  = json.loads(json_path.read_text(encoding='utf-8'))
    return model, info


# ── baselines: lookup por combo ──────────────────────────────────────────────

def cargar_baselines_combo(v_in, v_out, scope='comp'):
    """
    Devuelve un DataFrame con 1 fila por baseline disponible (buy_and_hold,
    trivial, lineal) para el (v_in, v_out) dado. Columnas:
        baseline, n_params, mae_train, mae_val, mae_test
    Si un CSV de baseline no existe o no tiene la combinación, se omite
    silenciosamente. Devuelve DataFrame vacío si no hay ninguno.

    Parámetros
    ----------
    scope : 'comp' | 'inv'
        'comp' (default): baselines de competición (log-retornos, split aleatorio).
        'inv'           : baselines de investigación (frac-diff, split temporal).
    """
    if scope == 'inv':
        files = BASELINE_FILES_INV
    elif scope == 'comp':
        files = BASELINE_FILES
    else:
        raise ValueError(f"scope debe ser 'comp' o 'inv'; recibido: {scope!r}")
    base_dir = Path(BASE_DRIVE) / '08_results' / 'tablas'
    filas    = []
    for key, fname in files.items():
        ruta = base_dir / fname
        if not ruta.exists():
            continue
        try:
            df = pd.read_csv(ruta)
            sel = df[(df['v_in'] == v_in) & (df['v_out'] == v_out)]
            if len(sel) == 0:
                continue
            r = sel.iloc[0]
            filas.append({
                'baseline':  key,
                'n_params':  int(r['n_params']),
                'mae_train': float(r['mae_train']),
                'mae_val':   float(r['mae_val']),
                'mae_test':  float(r['mae_test']),
            })
        except Exception:
            continue
    return pd.DataFrame(filas)


def resumen_vs_baselines(maes, v_in, v_out, nombre_modelo='modelo',
                         n_params=None, verbose=True):
    """
    Imprime una tabla comparando los MAE del modelo actual vs los 3 baselines
    para el (v_in, v_out) dado, con delta absoluto y % de mejora respecto al
    modelo. Devuelve el DataFrame.

    maes:     dict con keys 'train', 'val', 'test'.
    n_params: nº de parámetros del modelo (opcional). Si se pasa, se muestra
              en la fila del modelo; si no, se deja en blanco ('—').
    """
    scope = 'inv' if str(nombre_modelo).startswith('inv_') else 'comp'
    df_b = cargar_baselines_combo(v_in, v_out, scope=scope)
    fila_modelo = pd.DataFrame([{
        'baseline':  nombre_modelo,
        'n_params':  float(n_params) if n_params is not None else np.nan,
        'mae_train': maes['train'],
        'mae_val':   maes['val'],
        'mae_test':  maes['test'],
    }])
    df = pd.concat([fila_modelo, df_b], ignore_index=True)

    # % mejora del modelo vs cada baseline en test (positivo = el modelo es mejor)
    if len(df_b) > 0:
        df['mejora_test_%'] = np.where(
            df['baseline'] == nombre_modelo, np.nan,
            (df['mae_test'] - maes['test']) / df['mae_test'] * 100.0
        )

    if verbose:
        try:
            from IPython.display import display
            print(f'> Comparativa MAE — V_in={v_in}, V_out={v_out}')
            fmt = {'mae_train':     '{:.5f}',
                   'mae_val':       '{:.5f}',
                   'mae_test':      '{:.5f}',
                   'n_params':      '{:,.0f}'}
            if 'mejora_test_%' in df.columns:
                fmt['mejora_test_%'] = '{:+.2f}%'
            display(
                df.style
                  .format(fmt, na_rep='—')
                  .hide(axis='index')
            )
        except Exception:
            print(f'\n> Comparativa MAE — V_in={v_in}, V_out={v_out}')
            print(df.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    return df


# ── BestRunTracker ───────────────────────────────────────────────────────────
#
# Callback que durante el fit guarda la mejor época en best.keras (vía
# ModelCheckpoint interno) y al terminar:
#   - evalúa MAE train/val/test del modelo restaurado al best,
#   - escribe SIEMPRE last.json (auditoría de la última ejecución),
#   - si mejora el best.json previo (menor mae_val por defecto):
#         · pisa best.json
#         · pisa el CSV parcial 08_results/tablas/parciales/{...}.csv
#     en caso contrario, RESTAURA best.keras a la versión histórica anterior.
#   - hace append a history.csv (1 fila por ejecución).
#
# Hparams: se extraen automáticamente del modelo (config + optimizer.config)
# y se complementan con los que pase el usuario en hparams_extra.

def _hparams_desde_modelo(model):
    """Extrae arquitectura + optimizer config a un dict serializable."""
    try:
        arch = model.get_config()
    except Exception:
        arch = {}
    try:
        opt_cfg = model.optimizer.get_config()
    except Exception:
        opt_cfg = {}
    return {'arch': arch, 'optimizer': opt_cfg}


def _to_jsonable(obj):
    """Convierte recursivamente np.* a tipos Python para json.dump."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _make_best_run_tracker_class():
    """
    Construye la clase BestRunTracker heredando de keras.callbacks.Callback.
    Se hace lazy para que metrics_utils se pueda importar sin tensorflow.
    """
    from tensorflow import keras

    class BestRunTracker(keras.callbacks.Callback):
        def __init__(self, nombre_modelo, v_in, v_out, datasets,
                     hparams_extra=None, monitor='val_loss', auto_plot=False):
            """
            datasets: tupla (X_tr, Y_tr, X_val, Y_val, X_test, Y_test).
            hparams_extra: dict con hparams no estructurales (batch_size,
                patience, semilla, scaler, etc.) para incluir en best.json.
            monitor: métrica usada por el ModelCheckpoint interno.
            auto_plot: si True, llama a plot_curva() automáticamente cuando
                el run mejora el mae_val histórico.
            """
            super().__init__()
            self.nombre_modelo = nombre_modelo
            self.v_in          = v_in
            self.v_out         = v_out
            self.X_tr, self.Y_tr, self.X_val, self.Y_val, self.X_test, self.Y_test = datasets
            self.hparams_extra = dict(hparams_extra or {})
            self.monitor       = monitor
            self.auto_plot     = auto_plot

            self.run_dir              = _run_dir(nombre_modelo, v_in, v_out)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.best_keras_path      = self.run_dir / 'best.keras'
            self.best_json_path       = self.run_dir / 'best.json'
            self.last_json_path       = self.run_dir / 'last.json'
            self.history_csv          = self.run_dir / 'history.csv'
            self.epoch_history_csv    = self.run_dir / 'epoch_history.csv'

            # Backup del best.keras anterior por si la nueva run no mejora.
            self._backup_path = None
            if self.best_keras_path.exists():
                self._backup_path = self.run_dir / '_prev_best.keras.bak'
                shutil.copy2(self.best_keras_path, self._backup_path)

            self._tmp_keras_path   = self.run_dir / '_lastrun_best.keras'
            self._best_monitor_val = np.inf
            self._t_start          = None
            self.maes_last         = None
            self.improved          = False  # True si este run mejoró mae_val histórico

        def on_train_begin(self, logs=None):
            import time
            self._best_monitor_val = np.inf
            self._t_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current = logs.get(self.monitor, np.inf)
            if current < self._best_monitor_val:
                self._best_monitor_val = current
                self.model.save(str(self._tmp_keras_path))

        def on_train_end(self, logs=None):

            import time
            from tensorflow import keras

            duration_seconds = round(time.time() - self._t_start, 1) if self._t_start else -1

            # Cargar el modelo de la mejor época de ESTA run para evaluar.
            if self._tmp_keras_path.exists():
                model_best = keras.models.load_model(self._tmp_keras_path)
            else:
                # No se llegó a guardar (p. ej. 0 epochs); usar modelo actual.
                model_best = self.model

            maes = calc_mae_all(model_best, self.X_tr, self.Y_tr,
                                self.X_val, self.Y_val, self.X_test, self.Y_test)
            self.maes_last = maes

            # best_epoch / epochs_run a partir de history del modelo (si existe).
            try:
                vl = self.model.history.history.get('val_loss', [])
                epochs_run = len(vl)
                best_epoch = int(np.argmin(vl)) if vl else -1
            except Exception:
                epochs_run, best_epoch = -1, -1

            n_params = int(model_best.count_params())
            hparams  = _hparams_desde_modelo(model_best)
            hparams['extra'] = self.hparams_extra

            run_info = {
                'modelo':            self.nombre_modelo,
                'v_in':              int(self.v_in),
                'v_out':             int(self.v_out),
                'mae_train':         maes['train'],
                'mae_val':           maes['val'],
                'mae_test':          maes['test'],
                'n_params':          n_params,
                'best_epoch':        best_epoch,
                'epochs_run':        epochs_run,
                'duration_seconds':  duration_seconds,
                'hparams':           hparams,
                'timestamp':         datetime.now().isoformat(timespec='seconds'),
            }

            # 1) last.json siempre
            self.last_json_path.write_text(
                json.dumps(_to_jsonable(run_info), indent=2), encoding='utf-8'
            )

            # 2) Comparar con best.json previo (si existe).
            prev_best_val = None
            if self.best_json_path.exists():
                try:
                    prev = json.loads(self.best_json_path.read_text(encoding='utf-8'))
                    prev_best_val = float(prev.get('mae_val', np.inf))
                except Exception:
                    prev_best_val = None

            mejora = (prev_best_val is None) or (maes['val'] < prev_best_val)
            self.improved = bool(mejora)

            if mejora:
                # Pisar best.keras con la mejor época de esta run.
                shutil.copy2(self._tmp_keras_path, self.best_keras_path)
                run_info['improved_over_mae_val'] = prev_best_val
                self.best_json_path.write_text(
                    json.dumps(_to_jsonable(run_info), indent=2), encoding='utf-8'
                )
                # Pisar el CSV parcial con la mejor histórica.
                guardar_resultado_parcial(
                    {'modelo':    self.nombre_modelo,
                     'v_in':      self.v_in,
                     'v_out':     self.v_out,
                     'n_params':  n_params,
                     'mae_train': maes['train'],
                     'mae_val':   maes['val'],
                     'mae_test':  maes['test']},
                    self.nombre_modelo, self.v_in, self.v_out
                )
                if self.auto_plot:
                    try:
                        plot_curva(self.model.history, self.nombre_modelo,
                                   self.v_in, self.v_out, save=True,
                                   mostrar_baselines=True)
                    except Exception:
                        pass
            else:
                # Restaurar best.keras anterior (estaba en _backup).
                if self._backup_path is not None and self._backup_path.exists():
                    shutil.copy2(self._backup_path, self.best_keras_path)

            # 3) history.csv (append-only)
            overfit_ratio = round(maes['val'] / maes['train'], 4) if maes['train'] > 0 else np.nan
            fila = {
                'timestamp':        run_info['timestamp'],
                'mae_train':        maes['train'],
                'mae_val':          maes['val'],
                'mae_test':         maes['test'],
                'overfit_ratio':    overfit_ratio,
                'best_epoch':       best_epoch,
                'epochs_run':       epochs_run,
                'n_params':         n_params,
                'duration_seconds': duration_seconds,
                'mejora':           bool(mejora),
            }
            df_fila = pd.DataFrame([fila])
            if self.history_csv.exists():
                df_fila.to_csv(self.history_csv, mode='a', header=False, index=False)
            else:
                df_fila.to_csv(self.history_csv, index=False)

            # 4) epoch_history.csv (append-only, 1 fila por época)
            try:
                h  = self.model.history.history
                ts = run_info['timestamp']
                ep_rows = [
                    {'timestamp': ts, 'epoch': ep,
                     'loss': h['loss'][ep], 'val_loss': h['val_loss'][ep]}
                    for ep in range(len(h.get('loss', [])))
                ]
                if ep_rows:
                    df_ep = pd.DataFrame(ep_rows)
                    if self.epoch_history_csv.exists():
                        df_ep.to_csv(self.epoch_history_csv, mode='a', header=False, index=False)
                    else:
                        df_ep.to_csv(self.epoch_history_csv, index=False)
            except Exception:
                pass

            # Limpieza de archivos temporales.
            for p in (self._tmp_keras_path, self._backup_path):
                if p is not None and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except Exception:
                        pass

    return BestRunTracker


def BestRunTracker(*args, **kwargs):
    """
    Factory que construye la clase callback al primer uso (lazy import de
    tensorflow). Uso típico:

        tracker = BestRunTracker(MODELO, V_IN, V_OUT,
                    datasets=(X_tr, Y_tr, X_val, Y_val, X_test, Y_test),
                    hparams_extra={'batch_size': BATCH_SIZE, 'patience': PATIENCE},
                    auto_plot=True)   # opcional: guarda curva automáticamente si mejora
        model.fit(..., callbacks=[tracker])
    """
    cls = _make_best_run_tracker_class()
    return cls(*args, **kwargs)
