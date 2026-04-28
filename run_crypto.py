"""
AutoPrompt-Quant — Cryptocurrency experiment.

Data:    Binance REST API (api_key required; no synthetic fallback).
Market:  Top 10 crypto assets by market cap (BTC, ETH, BNB, XRP, ADA,
         SOL, DOT, AVAX, LINK, DOGE) — USDT pairs on Binance spot.
IS:      2020-01-01 → 2022-12-31  (3 years; most coins listed from 2020)
OOS:     4 × 6-month windows, 2023-01-01 → 2024-12-31
TC:      10 bps one-way  (spot exchange taker fee + spread)
Band:    None  (no circuit-breaker; crypto trades 24/7)
Ann:     365 calendar days/year  (continuous 24/7 trading)

Run:
    python run_crypto.py
"""

import os, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.shared import run_pipeline
from data.fetchers import fetch_crypto, CRYPTO_SYMBOLS

# ── API key — loaded from config.py (gitignored) or env var ──────────────
try:
    from config import BINANCE_API_KEY
except ImportError:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')

# ── Market config ─────────────────────────────────────────────────────────
MARKET     = 'Crypto'
IS_START   = '2020-01-01'
IS_END     = '2022-12-31'
TC         = 0.0010          # 10 bps one-way
PRICE_BAND = None
ANN        = 365             # crypto trades every calendar day
N_PROMPTS  = 50

WF_WINDOWS = [
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
]

# ── LLM prompts ────────────────────────────────────────────────────────────
META_SYSTEM = (
    'You are a quant research assistant for cryptocurrency markets.\n'
    'Generate a concise prompt (150-250 words) for a coding-agent LLM to implement '
    'a long-only crypto trading strategy.\n'
    'The strategy must be implementable as:\n'
    '    def gen_position(df: pd.DataFrame, **params) -> pd.Series\n'
    'where df has columns [open,high,low,close,volume] and output is {{0,1}} binary series.\n'
    'Market context: 24/7 continuous trading (365 calendar days, no weekends off), '
    'high annualised volatility (50-120%% typical), pronounced momentum and mean-reversion '
    'regimes, whale-driven volume spikes, exchange taker fee c=0.10%% one-way. '
    'No short selling or leverage modelled.\n'
    'Incorporate: signal_type={signal_type}, holding_period={holding_period}, '
    'risk_framing={risk_framing}, complexity={complexity}.\n'
    'Output ONLY the prompt text — no preamble, no commentary.'
)

CODING_SYSTEM = (
    'You are an expert Python quant developer for cryptocurrency markets '
    '(BTC, ETH, BNB, SOL, etc.).\n'
    'Implement the trading strategy described below.\n'
    'STRICT REQUIREMENTS:\n'
    '  - First line must declare: PARAM_GRID = {"lookback": [5,10,20], "threshold": [0.0,0.01,0.02]}\n'
    '  - Then: import pandas as pd\\nimport numpy as np\n'
    '  - Define exactly ONE function: def gen_position(df: pd.DataFrame, **params) -> pd.Series\n'
    '  - df columns: open, high, low, close, volume (float, DatetimeIndex, DAILY bars)\n'
    '  - Crypto data has 365 rows per year — consecutive daily data including weekends\n'
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
    base_dir   = ROOT / 'results' / 'crypto'
    cache_path = base_dir / 'data' / 'crypto_data.pkl'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    price_data = fetch_crypto(
        cache_path = cache_path,
        start      = '2019-01-01',   # extra year for indicator warm-up
        end        = '2024-12-31',
        symbols    = CRYPTO_SYMBOLS,
        api_key    = BINANCE_API_KEY,
    )

    run_pipeline(
        market        = MARKET,
        symbols       = CRYPTO_SYMBOLS,
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
