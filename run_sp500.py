"""
AutoPrompt-Quant — S&P 500 (US large-cap equity) experiment.

Data:    yfinance (free, no API key required).
Market:  Top 30 S&P 500 constituents by market cap (as of 2024).
IS:      2018-01-01 → 2020-12-31  (same as VN30 for cross-market comparability)
OOS:     8 × 6-month windows, 2021-01-01 → 2024-12-31
TC:      5 bps one-way  (liquid US equities, tight spreads)
Band:    None (no circuit-breaker modelled in backtest)
Ann:     252 trading days/year.

Run:
    python run_sp500.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.shared import run_pipeline
from data.fetchers import fetch_sp500, SP500_SYMBOLS

# ── Market config ─────────────────────────────────────────────────────────
MARKET     = 'SP500'
IS_START   = '2018-01-01'
IS_END     = '2020-12-31'
TC         = 0.0005          # 5 bps one-way
PRICE_BAND = None
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

# ── LLM prompts ────────────────────────────────────────────────────────────
META_SYSTEM = (
    'You are a quant research assistant for US large-cap equity markets (S&P 500).\n'
    'Generate a concise prompt (150-250 words) for a coding-agent LLM to implement '
    'a long-only S&P 500 trading strategy.\n'
    'The strategy must be implementable as:\n'
    '    def gen_position(df: pd.DataFrame, **params) -> pd.Series\n'
    'where df has columns [open,high,low,close,volume] and output is {{0,1}} binary series.\n'
    'Market context: NYSE/NASDAQ business-day calendar, highly liquid blue-chip stocks, '
    'split- and dividend-adjusted close prices, efficient pricing with macro-driven rotations, '
    'brokerage cost c=0.05%% one-way.\n'
    'Incorporate: signal_type={signal_type}, holding_period={holding_period}, '
    'risk_framing={risk_framing}, complexity={complexity}.\n'
    'Output ONLY the prompt text — no preamble, no commentary.'
)

CODING_SYSTEM = (
    'You are an expert Python quant developer for US large-cap equities (S&P 500).\n'
    'Implement the trading strategy described below.\n'
    'STRICT REQUIREMENTS:\n'
    '  - First line must declare: PARAM_GRID = {"lookback": [5,10,20], "threshold": [0.0,0.01,0.02]}\n'
    '  - Then: import pandas as pd\\nimport numpy as np\n'
    '  - Define exactly ONE function: def gen_position(df: pd.DataFrame, **params) -> pd.Series\n'
    '  - df columns: open, high, low, close, volume (float, DatetimeIndex, NYSE business days)\n'
    '  - close is split- and dividend-adjusted\n'
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
    base_dir   = ROOT / 'results' / 'sp500'
    cache_path = base_dir / 'data' / 'sp500_data.pkl'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    price_data = fetch_sp500(
        cache_path = cache_path,
        start      = '2017-01-01',
        end        = '2024-12-31',
        symbols    = SP500_SYMBOLS,
    )

    run_pipeline(
        market        = MARKET,
        symbols       = SP500_SYMBOLS,
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
