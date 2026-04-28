"""
AutoPrompt-Quant — VN30 (Vietnamese equity) experiment.

Data:    QuantVN API (falls back to synthetic OHLCV if API key absent).
Market:  30 VN30 index constituents on HOSE.
IS:      2018-01-01 → 2020-12-31
OOS:     8 × 6-month windows, 2021-01-01 → 2024-12-31
TC:      15 bps one-way  (VN30 retail brokerage, paper §IV-A)
Band:    ±7% daily price limit (HOSE rule, paper §III-B)
Ann:     252 trading days/year.

Run:
    python run_vn30.py
"""

import os, sys, shutil
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.shared import run_pipeline
from data.fetchers import fetch_vn30, VN30_SYMBOLS

# ── API key — loaded from config.py (gitignored) or env var ──────────────
try:
    from config import QUANTVN_API_KEY
except ImportError:
    QUANTVN_API_KEY = os.environ.get('QUANTVN_API_KEY', '')

# ── Market config ─────────────────────────────────────────────────────────
MARKET     = 'VN30'
IS_START   = '2018-01-01'
IS_END     = '2020-12-31'
TC         = 0.0015          # 15 bps one-way
PRICE_BAND = 0.07            # ±7% daily limit
ANN        = 252
N_PROMPTS  = 50

WF_WINDOWS = [
    ('2021-01-01', '2021-06-30'),
    ('2021-07-01', '2021-12-31'),
    ('2022-01-01', '2022-06-30'),
    ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
]

# ── LLM prompts — aligned to paper §III-A ────────────────────────────────
META_SYSTEM = (
    'You are a quant research assistant for Vietnamese equity markets (VN30/HOSE).\n'
    'Generate a concise prompt (150-250 words) for a coding-agent LLM to implement '
    'a VN30 trading strategy.\n'
    'The strategy must be implementable as:\n'
    '    def gen_position(df: pd.DataFrame, **params) -> pd.Series\n'
    'where df has columns [open,high,low,close,volume] and output is {{0,1}} binary series.\n'
    'Market constraints to encode: T+2 settlement (minimum 2-day holding), '
    '±7% HOSE daily price band (positions clipped if limit hit), '
    'long/flat only (no short selling), retail brokerage c=0.15%% one-way.\n'
    'Incorporate: signal_type={signal_type}, holding_period={holding_period}, '
    'risk_framing={risk_framing}, complexity={complexity}.\n'
    'Output ONLY the prompt text — no preamble, no commentary.'
)

CODING_SYSTEM = (
    'You are an expert Python quant developer for VN30 Vietnamese equities.\n'
    'Implement the trading strategy described below.\n'
    'STRICT REQUIREMENTS:\n'
    '  - First line must declare: PARAM_GRID = {"lookback": [5,10,20], "threshold": [0.0,0.01,0.02]}\n'
    '  - Then: import pandas as pd\\nimport numpy as np\n'
    '  - Define exactly ONE function: def gen_position(df: pd.DataFrame, **params) -> pd.Series\n'
    '  - df columns: open, high, low, close, volume (float, DatetimeIndex, business days)\n'
    '  - Output: pd.Series of ONLY 0 or 1 (integers). 1=long, 0=flat. NEVER use -1.\n'
    '  - Same length and index as df\n'
    '  - Parameters: lookback=params.get("lookback",10), threshold=params.get("threshold",0.01)\n'
    '  - No look-ahead bias: use shift() before comparing signals\n'
    '  - Fill NaN/leading rows with 0 using .fillna(0)\n'
    '  - pandas and numpy only — no external libraries\n'
    '  - Return ONLY a Python code block enclosed in ```python ... ```\n'
    'PARAM_GRID at evaluation: lookback=[5,10,20], threshold=[0.0,0.01,0.02]\n'
)

# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    base_dir = ROOT / 'results' / 'vn30'
    ckpt_dir = base_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Reuse existing VN30 checkpoints from the original pipeline run
    # so we don't spend hours re-running LLM calls already completed.
    legacy_ckpt = ROOT.parent / 'ckpt_ollama'
    if legacy_ckpt.exists():
        for pkl in legacy_ckpt.glob('*.pkl'):
            dst = ckpt_dir / pkl.name
            if not dst.exists():
                shutil.copy(pkl, dst)
                print(f'[VN30] Copied checkpoint: {pkl.name}')
        for csv in legacy_ckpt.glob('*.csv'):
            dst = ckpt_dir / csv.name
            if not dst.exists():
                shutil.copy(csv, dst)
                print(f'[VN30] Copied CSV: {csv.name}')

    this_cache = ckpt_dir / 'price_data.pkl'   # already copied above if exists
    price_data = fetch_vn30(
        cache_path = this_cache,
        start      = '2017-01-01',
        end        = '2024-12-31',
        symbols    = VN30_SYMBOLS,
        api_key    = QUANTVN_API_KEY,
    )

    run_pipeline(
        market        = MARKET,
        symbols       = VN30_SYMBOLS,
        price_data    = price_data,
        is_start      = IS_START,
        is_end        = IS_END,
        wf_windows    = WF_WINDOWS,
        tc            = TC,
        price_band    = PRICE_BAND,
        ann           = ANN,
        meta_system   = META_SYSTEM,
        coding_system = CODING_SYSTEM,
        base_dir      = base_dir,
        n_prompts     = N_PROMPTS,
    )
