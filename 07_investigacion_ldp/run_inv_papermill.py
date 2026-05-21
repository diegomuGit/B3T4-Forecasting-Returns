"""
Ejecuta secuencialmente los 10 notebooks `inv_*_vout90.ipynb` con papermill.

- Sobrescribe cada .ipynb in-place (outputs quedan embebidos en el propio fichero).
- Si un notebook falla, captura la excepcion, imprime el traceback y continua con el siguiente.
- Al final imprime un resumen OK/KO.

Cada notebook ya se encarga de persistir su curva, parcial CSV y checkpoint
via `BestRunTracker` + `plot_curva` + `ModelCheckpoint`. Este script solo
los lanza en orden.

Uso:
    python run_inv_papermill.py
"""

from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path

import papermill as pm

HERE = Path(__file__).resolve().parent

NOTEBOOKS = [
    'inv_rnn_vin5_vout90.ipynb',
    'inv_cnn_vin5_vout90.ipynb',
    'inv_mlp_vin10_vout90.ipynb',
    'inv_rnn_vin10_vout90.ipynb',
    'inv_mixto_vin10_vout90.ipynb',
    'inv_mlp_vin30_vout90.ipynb',
    'inv_cnn_vin30_vout90.ipynb',
    'inv_mixto_vin30_vout90.ipynb',
    'inv_rnn_vin90_vout90.ipynb',
    'inv_cnn_vin90_vout90.ipynb',
]


def main():
    resultados = []
    t_total = time.time()
    for i, nb in enumerate(NOTEBOOKS, start=1):
        ruta = HERE / nb
        if not ruta.exists():
            print(f'[{i:2d}/{len(NOTEBOOKS)}] SKIP {nb} (no existe)')
            resultados.append((nb, 'SKIP', 0.0))
            continue

        print(f'\n[{i:2d}/{len(NOTEBOOKS)}] ===> {nb}')
        print(f'         inicio: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        t0 = time.time()
        try:
            pm.execute_notebook(
                input_path=str(ruta),
                output_path=str(ruta),
                kernel_name='python3',
                progress_bar=False,
            )
            estado = 'OK'
        except Exception as e:
            estado = f'KO ({e.__class__.__name__})'
            traceback.print_exc()
        dt = time.time() - t0
        print(f'         fin   : {time.strftime("%Y-%m-%d %H:%M:%S")}  '
              f'duracion: {dt/60:.1f} min  [{estado}]')
        resultados.append((nb, estado, dt))

    dt_total = time.time() - t_total
    ok = sum(1 for _, e, _ in resultados if e == 'OK')
    ko = sum(1 for _, e, _ in resultados if e.startswith('KO'))
    sk = sum(1 for _, e, _ in resultados if e == 'SKIP')
    print('\n' + '=' * 60)
    print(f'RESUMEN  ({dt_total/60:.1f} min totales)')
    print('=' * 60)
    for nb, estado, dt in resultados:
        print(f'  [{estado:>20s}]  {nb}  ({dt/60:.1f} min)')
    print('-' * 60)
    print(f'  OK={ok}   KO={ko}   SKIP={sk}   TOTAL={len(NOTEBOOKS)}')

    sys.exit(0 if ko == 0 else 1)


if __name__ == '__main__':
    main()
