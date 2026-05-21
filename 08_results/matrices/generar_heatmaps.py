"""
Generador de heatmaps para el Taller B3-T4.

Produce 5 PNGs + 1 CSV en 08_results/matrices/:
  - 4 imagenes combinadas (una por familia): train | val | test en columnas,
    escala de color compartida y una unica leyenda.
  - 1 matriz de competicion 4x4 con gradiente de MAE test.
  - 1 CSV consolidado con los 64 resultados.

Ejecutar:
    python 08_results/matrices/generar_heatmaps.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE = Path(__file__).resolve().parents[2]
PARCIALES = BASE / '08_results' / 'tablas' / 'parciales'
SALIDA = BASE / '08_results' / 'matrices'

FAMILIA_DE = {
    'mlp':      'densas',
    'rnn':      'recurrente',
    'lstm_gru': 'recurrente',
    'cnn':      'convolucional',
    'conv1d':   'convolucional',
    'mixto':    'mixto',
}
FAMILIA_DE_INV = {
    'inv_mlp':   'densas',
    'inv_rnn':   'recurrente',
    'inv_cnn':   'convolucional',
    'inv_mixto': 'mixto',
}
FAMILIAS = ['densas', 'recurrente', 'convolucional', 'mixto']
SPLITS   = ['train', 'val', 'test']
V_INS  = [5, 10, 30, 90]
V_OUTS = [1, 5, 30, 90]


def cargar_resultados() -> pd.DataFrame:
    archivos = [f for f in PARCIALES.glob('*.csv') if not f.name.startswith('inv_')]
    if not archivos:
        raise FileNotFoundError(f'No se encontraron CSV en {PARCIALES}')

    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    df['modelo'] = df['modelo'].str.lower()
    df['familia'] = df['modelo'].map(FAMILIA_DE)

    if df['familia'].isna().any():
        desconocidos = df.loc[df['familia'].isna(), 'modelo'].unique().tolist()
        raise ValueError(f'Modelos sin familia mapeada: {desconocidos}')

    conteos = df.groupby('familia').size()
    for fam in FAMILIAS:
        n = int(conteos.get(fam, 0))
        if n != 16:
            raise ValueError(f'Familia "{fam}" tiene {n} filas, esperaba 16.')

    dup = df.duplicated(subset=['familia', 'v_in', 'v_out'])
    if dup.any():
        raise ValueError(f'Combos duplicados por familia:\n{df[dup]}')

    return df


def _figura_triple(titulo: str, paneles, cmap, vmin, vmax, cbar_label):
    """
    paneles: lista de 3 dicts con keys 'split_label', 'pivot_mae', 'annot'.
    Genera figura con 3 heatmaps en fila, escala compartida y una sola leyenda
    (sin solaparse con los axes gracias a constrained_layout).
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(
        1, 3, figsize=(22, 7),
        gridspec_kw={'width_ratios': [1, 1, 1]},
        layout='constrained',
    )
    fig.suptitle(titulo, fontsize=14)

    for ax, panel in zip(axes, paneles):
        sns.heatmap(
            panel['pivot_mae'],
            annot=panel['annot'], fmt='',
            cmap=cmap, norm=norm,
            ax=ax,
            linewidths=0.5,
            cbar=False,
            annot_kws={'fontsize': 9},
        )
        ax.set_title(panel['split_label'])
        ax.set_xlabel('V_out (dias de salida)')
        ax.set_ylabel('V_in (dias de entrada)' if ax is axes[0] else '')

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    return fig


def heatmap_familia_combinado(df: pd.DataFrame, familia: str) -> Path:
    sub = df[df['familia'] == familia]

    pivot_params = (
        sub.pivot(index='v_in', columns='v_out', values='n_params')
           .reindex(index=V_INS, columns=V_OUTS)
    )

    paneles = []
    pivots_mae = []
    for split in SPLITS:
        pivot_mae = (sub.pivot(index='v_in', columns='v_out', values=f'mae_{split}')
                        .reindex(index=V_INS, columns=V_OUTS))
        annot = np.empty_like(pivot_mae.values, dtype=object)
        for i in range(pivot_mae.shape[0]):
            for j in range(pivot_mae.shape[1]):
                mae = pivot_mae.values[i, j]
                n   = pivot_params.values[i, j]
                annot[i, j] = f'{mae:.5f}\nparams={int(n):,}'
        paneles.append({'split_label': f'MAE {split}', 'pivot_mae': pivot_mae, 'annot': annot})
        pivots_mae.append(pivot_mae)

    vmin = min(p.values.min() for p in pivots_mae)
    vmax = max(p.values.max() for p in pivots_mae)

    fig = _figura_triple(
        titulo=f'{familia.capitalize()} — MAE (train / val / test)',
        paneles=paneles,
        cmap=mpl.cm.YlOrRd,
        vmin=vmin, vmax=vmax,
        cbar_label='MAE',
    )

    ruta = SALIDA / f'heatmap_{familia}_combinado.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'> {ruta}')
    return ruta


def matriz_competicion(df: pd.DataFrame) -> Path:
    paneles = []
    pivots_mae = []
    for split in SPLITS:
        col = f'mae_{split}'
        idx = df.groupby(['v_in', 'v_out'])[col].idxmin()
        ganadores = df.loc[idx].reset_index(drop=True)
        pivot_mae = (ganadores.pivot(index='v_in', columns='v_out', values=col)
                              .reindex(index=V_INS, columns=V_OUTS))
        pivot_fam = (ganadores.pivot(index='v_in', columns='v_out', values='familia')
                              .reindex(index=V_INS, columns=V_OUTS))
        annot = np.empty_like(pivot_mae.values, dtype=object)
        for i in range(pivot_mae.shape[0]):
            for j in range(pivot_mae.shape[1]):
                mae = pivot_mae.values[i, j]
                fam = pivot_fam.values[i, j]
                annot[i, j] = f'{mae:.5f}\n{fam}'
        paneles.append({'split_label': f'Mejor MAE {split}', 'pivot_mae': pivot_mae, 'annot': annot})
        pivots_mae.append(pivot_mae)

    vmin = min(p.values.min() for p in pivots_mae)
    vmax = max(p.values.max() for p in pivots_mae)

    fig = _figura_triple(
        titulo='Competicion — Mejor MAE por combo (nombre = familia ganadora)',
        paneles=paneles,
        cmap=mpl.cm.YlGn,
        vmin=vmin, vmax=vmax,
        cbar_label='MAE (mas oscuro = peor)',
    )

    ruta = SALIDA / 'matriz_competicion_combinado.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'> {ruta}')
    return ruta


def matriz_competicion_test(df: pd.DataFrame) -> Path:
    idx = df.groupby(['v_in', 'v_out'])['mae_test'].idxmin()
    ganadores = df.loc[idx].reset_index(drop=True)

    pivot_mae = (ganadores.pivot(index='v_in', columns='v_out', values='mae_test')
                          .reindex(index=V_INS, columns=V_OUTS))
    pivot_familia = (ganadores.pivot(index='v_in', columns='v_out', values='familia')
                              .reindex(index=V_INS, columns=V_OUTS))

    annot = np.empty_like(pivot_mae.values, dtype=object)
    for i in range(pivot_mae.shape[0]):
        for j in range(pivot_mae.shape[1]):
            mae = pivot_mae.values[i, j]
            fam = pivot_familia.values[i, j]
            annot[i, j] = f'{mae:.5f}\n{fam}'

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        pivot_mae,
        annot=annot, fmt='',
        cmap='YlGn',
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'MAE test (mas oscuro = peor)'},
        annot_kws={'fontsize': 9, 'weight': 'bold'},
    )
    ax.set_title('Competicion — Mejor MAE test por combo\n(nombre = familia ganadora)')
    ax.set_xlabel('V_out (dias de salida)')
    ax.set_ylabel('V_in (dias de entrada)')
    fig.tight_layout()

    ruta = SALIDA / 'matriz_competicion_test.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'> {ruta}')
    return ruta


def cargar_inv() -> pd.DataFrame:
    archivos = sorted(PARCIALES.glob('inv_*.csv'))
    if len(archivos) != 16:
        raise ValueError(f'Esperaba 16 ficheros inv_*, encontrados {len(archivos)}.')

    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    df['modelo'] = df['modelo'].str.lower()
    df['familia'] = df['modelo'].map(FAMILIA_DE_INV)

    if df['familia'].isna().any():
        desconocidos = df.loc[df['familia'].isna(), 'modelo'].unique().tolist()
        raise ValueError(f'Modelos inv_ sin familia mapeada: {desconocidos}')

    if not (df['v_out'] == 90).all():
        raise ValueError('Algun fichero inv_* no tiene v_out=90.')

    conteos = df.groupby('familia').size()
    for fam in FAMILIAS:
        n = int(conteos.get(fam, 0))
        if n != 4:
            raise ValueError(f'Familia inv "{fam}" tiene {n} filas, esperaba 4.')

    return df


def comparativa_inv_vs_original(df_orig: pd.DataFrame, df_inv: pd.DataFrame) -> Path:
    orig_v90 = df_orig[df_orig['v_out'] == 90]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparativa inv vs original — MAE test (v_out=90)', fontsize=14, y=0.995)

    x = np.arange(len(V_INS))
    w = 0.38

    for ax, familia in zip(axes.flat, FAMILIAS):
        s_orig = (orig_v90[orig_v90['familia'] == familia]
                  .set_index('v_in')['mae_test'].reindex(V_INS))
        s_inv  = (df_inv[df_inv['familia'] == familia]
                  .set_index('v_in')['mae_test'].reindex(V_INS))

        b1 = ax.bar(x - w/2, s_orig.values, w, label='original', color='#1f77b4')
        b2 = ax.bar(x + w/2, s_inv.values,  w, label='inv',      color='#ff7f0e')

        for bars in (b1, b2):
            ax.bar_label(bars, fmt='%.5f', fontsize=8, padding=2)

        ax.set_title(familia.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels([f'v_in={v}' for v in V_INS])
        ax.set_ylabel('MAE test')
        ax.grid(True, axis='y', alpha=0.3)
        ymax = np.nanmax([s_orig.values, s_inv.values]) * 1.18
        ax.set_ylim(0, ymax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    ruta = SALIDA / 'comparativa_inv_vs_original.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'> {ruta}')
    return ruta


def generar_tabla_consolidada(df: pd.DataFrame) -> Path:
    tabla = (
        df[['modelo', 'v_in', 'v_out', 'n_params', 'mae_train', 'mae_val', 'mae_test']]
        .rename(columns={'modelo': 'model_name', 'v_in': 'input_window', 'v_out': 'output_window'})
        .sort_values(['model_name', 'input_window', 'output_window'])
        .reset_index(drop=True)
    )
    ruta = SALIDA / 'resultados_consolidados.csv'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    print(f'> {ruta}')
    return ruta


def main() -> None:
    df = cargar_resultados()
    for familia in FAMILIAS:
        heatmap_familia_combinado(df, familia)
    matriz_competicion(df)
    matriz_competicion_test(df)
    df_inv = cargar_inv()
    comparativa_inv_vs_original(df, df_inv)
    generar_tabla_consolidada(df)
    print(f'\n> 7 PNGs + 1 CSV generados en {SALIDA}')


if __name__ == '__main__':
    main()
