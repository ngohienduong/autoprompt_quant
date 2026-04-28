"""
AutoPrompt-Quant — one-shot data prefetch script.

Run this BEFORE any experiment runner to download and cache all three
market datasets.  Subsequent calls to run_vn30.py / run_sp500.py /
run_crypto.py will load from cache and skip network requests entirely.

Usage
-----
    python fetch_data.py              # fetch all three markets
    python fetch_data.py sp500        # SP500 only  (no API key needed)
    python fetch_data.py vn30 crypto  # VN30 + Crypto

Data sources
------------
  VN30   — QuantVN API    (requires QUANTVN_API_KEY in config.py or env)
  SP500  — yfinance       (free, no API key)
  Crypto — Binance REST   (requires BINANCE_API_KEY in config.py or env)

Output
------
  results/vn30/checkpoints/price_data.pkl      (~5 MB)
  results/sp500/data/sp500_data.pkl            (~8 MB)
  results/crypto/data/crypto_data.pkl          (~2 MB)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── API keys: config.py takes priority, fall back to env vars ─────────────
try:
    from config import QUANTVN_API_KEY
except ImportError:
    QUANTVN_API_KEY = os.environ.get('QUANTVN_API_KEY', '')

try:
    from config import BINANCE_API_KEY
except ImportError:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')

from data.fetchers import (
    fetch_vn30,    VN30_SYMBOLS,
    fetch_sp500,   SP500_SYMBOLS,
    fetch_crypto,  CRYPTO_SYMBOLS,
)


def fetch_vn30_data():
    cache = ROOT / 'results' / 'vn30' / 'checkpoints' / 'price_data.pkl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not QUANTVN_API_KEY:
        print('[VN30] WARNING: QUANTVN_API_KEY not set — skipping VN30.')
        print('       Set it in config.py or export QUANTVN_API_KEY=<key>')
        return False
    print('[VN30] Fetching 30 symbols via QuantVN (2017-01-01 → 2024-12-31)...')
    data = fetch_vn30(
        cache_path=cache,
        start='2017-01-01',
        end='2024-12-31',
        symbols=VN30_SYMBOLS,
        api_key=QUANTVN_API_KEY,
    )
    print(f'[VN30] Done — {len(data)} symbols cached at {cache}\n')
    return True


def fetch_sp500_data():
    cache = ROOT / 'results' / 'sp500' / 'data' / 'sp500_data.pkl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    print('[SP500] Fetching 30 symbols via yfinance (2017-01-01 → 2024-12-31)...')
    data = fetch_sp500(
        cache_path=cache,
        start='2017-01-01',
        end='2024-12-31',
        symbols=SP500_SYMBOLS,
    )
    print(f'[SP500] Done — {len(data)} symbols cached at {cache}\n')
    return True


def fetch_crypto_data():
    cache = ROOT / 'results' / 'crypto' / 'data' / 'crypto_data.pkl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not BINANCE_API_KEY:
        print('[Crypto] WARNING: BINANCE_API_KEY not set — skipping Crypto.')
        print('         Set it in config.py or export BINANCE_API_KEY=<key>')
        return False
    print('[Crypto] Fetching 10 symbols via Binance REST (2019-01-01 → 2024-12-31)...')
    data = fetch_crypto(
        cache_path=cache,
        start='2019-01-01',
        end='2024-12-31',
        symbols=CRYPTO_SYMBOLS,
        api_key=BINANCE_API_KEY,
    )
    print(f'[Crypto] Done — {len(data)} symbols cached at {cache}\n')
    return True


FETCHERS = {
    'vn30':   fetch_vn30_data,
    'sp500':  fetch_sp500_data,
    'crypto': fetch_crypto_data,
}

if __name__ == '__main__':
    targets = [a.lower() for a in sys.argv[1:]] or list(FETCHERS)
    unknown = [t for t in targets if t not in FETCHERS]
    if unknown:
        print(f'Unknown market(s): {unknown}. Choose from: {list(FETCHERS)}')
        sys.exit(1)

    print('=' * 60)
    print('AutoPrompt-Quant — data prefetch')
    print(f'Markets: {targets}')
    print('=' * 60 + '\n')

    results = {}
    for market in targets:
        try:
            results[market] = FETCHERS[market]()
        except Exception as e:
            print(f'[{market.upper()}] FAILED: {e}\n')
            results[market] = False

    print('=' * 60)
    print('Summary:')
    for market, ok in results.items():
        status = 'OK' if ok else 'SKIPPED / FAILED'
        print(f'  {market.upper():8s} {status}')
    print('=' * 60)

    failed = [m for m, ok in results.items() if not ok]
    if failed:
        print(f'\nFix the issues above, then re-run: python fetch_data.py {" ".join(failed)}')
        sys.exit(1)
    else:
        print('\nAll data cached. You can now run: python run_all.py')
