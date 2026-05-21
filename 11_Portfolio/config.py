"""
config.py — Central configuration for the portfolio backtest.

All tunable parameters live here. Change them once; every module reads from this file.
Designed to be imported as:  from config import CFG

To switch models, change ACTIVE_MODEL before importing or after:
    CFG.ACTIVE_MODEL = 'inv_rnn'
    CFG.apply_model()   # updates V_IN, MODELO, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Model catalogue ──────────────────────────────────────────────────
# Each entry: (MODELO prefix, V_IN, V_OUT, architecture type)
# Architecture types: 'cnn', 'mlp', 'rnn', 'mixto'
MODEL_CATALOG = {
    'inv_cnn': {
        'modelo':  'inv_cnn',
        'v_in':    10,
        'v_out':   90,
        'arch':    'cnn',
    },
    'inv_mlp': {
        'modelo':  'inv_mlp',
        'v_in':    5,
        'v_out':   90,
        'arch':    'mlp',
    },
    'inv_rnn': {
        'modelo':  'inv_rnn',
        'v_in':    30,
        'v_out':   90,
        'arch':    'rnn',
    },
    'inv_mixto': {
        'modelo':  'inv',       # MODELO='Inv' → .lower() = 'inv'
        'v_in':    90,
        'v_out':   90,
        'arch':    'mixto',
    },
}


@dataclass
class PortfolioConfig:
    """Single source of truth for every backtest parameter."""

    # ── Asset universe ────────────────────────────────────────────────
    TICKERS: List[str] = field(default_factory=lambda: [
        'AEP', 'BA', 'CAT', 'CNP', 'CVX',
        'DIS', 'DTE', 'ED', 'GD', 'GE',
        'HON', 'HPQ', 'IBM', 'IP', 'JNJ',
        'KO', 'KR', 'MMM', 'MO', 'MRK',
        'MSI', 'PG', 'XOM',
    ])

    @property
    def N_ASSETS(self) -> int:
        return len(self.TICKERS)

    # ── Active model (change this to switch) ──────────────────────────
    # Options: 'inv_cnn', 'inv_mlp', 'inv_rnn', 'inv_mixto'
    ACTIVE_MODEL: str = 'inv_cnn'

    # ── Model / frac-diff settings (updated by apply_model) ──────────
    MODELO: str  = 'inv_cnn'
    V_IN:   int  = 10
    V_OUT:  int  = 90
    ARCH:   str  = 'cnn'
    D_FRAC: float = 0.40
    FFD_THRESHOLD: float = 1e-5
    SCALER: str  = 'standard'

    # ── Strategy parameters ───────────────────────────────────────────
    TOP_K: int = 5
    REBAL_DAYS: int = 90
    CASH_RATE: float = 0.0
    ENABLE_TAKE_PROFIT: bool = True

    # ── Backtest period ───────────────────────────────────────────────
    BT_START: str = '2025-01-01'
    BT_END:   str = '2025-12-31'

    # ── Data lookback ─────────────────────────────────────────────────
    LOOKBACK_YEARS: int = 10

    # ── Paths ─────────────────────────────────────────────────────────
    CKPT_REL_PATH: str = '08_results/checkpoints/{modelo}_vin{v_in}_vout{v_out}_best.weights.h5'

    def ckpt_path(self, base: str) -> str:
        import os
        rel = self.CKPT_REL_PATH.format(
            modelo=self.MODELO, v_in=self.V_IN, v_out=self.V_OUT
        )
        return os.path.join(base, rel)

    def apply_model(self, model_key: str = None):
        """Apply settings from MODEL_CATALOG for the given (or active) model."""
        key = model_key or self.ACTIVE_MODEL
        if key not in MODEL_CATALOG:
            raise ValueError(
                f'Unknown model "{key}". Available: {list(MODEL_CATALOG.keys())}'
            )
        entry = MODEL_CATALOG[key]
        self.ACTIVE_MODEL = key
        self.MODELO = entry['modelo']
        self.V_IN   = entry['v_in']
        self.V_OUT  = entry['v_out']
        self.ARCH   = entry['arch']
        # REBAL_DAYS should match V_OUT
        self.REBAL_DAYS = entry['v_out']

    def __post_init__(self):
        """Auto-apply the default active model on creation."""
        self.apply_model()


# ── Default instance ──────────────────────────────────────────────────
CFG = PortfolioConfig()
