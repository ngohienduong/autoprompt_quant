"""
AutoPrompt-Quant — master runner.

Runs all three market experiments sequentially:
    1. VN30  (Vietnamese equities)
    2. SP500 (US large-cap equities)
    3. Crypto (top-10 by market cap)

Each market writes results to results/<market>/:
    checkpoints/   — pkl files (resumable)
    figures/       — PNG plots
    <market>.log   — full experiment log
    oos_summary.csv, cross_model.csv

Usage:
    python run_all.py              # run everything
    python run_all.py sp500 crypto # run specific markets
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

RUNNERS = {
    'vn30':   ROOT / 'run_vn30.py',
    'sp500':  ROOT / 'run_sp500.py',
    'crypto': ROOT / 'run_crypto.py',
}

def main():
    targets = sys.argv[1:] or list(RUNNERS.keys())
    unknown = [t for t in targets if t not in RUNNERS]
    if unknown:
        print(f'Unknown market(s): {unknown}. Choose from: {list(RUNNERS.keys())}')
        sys.exit(1)

    for mkt in targets:
        script = RUNNERS[mkt]
        print(f'\n{"="*60}')
        print(f'  Running: {mkt.upper()}  ({script.name})')
        print(f'{"="*60}\n')
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            print(f'\n[WARNING] {mkt} exited with code {result.returncode}. '
                  'Continuing with next market...\n')

    print('\nAll markets complete. Results in: results/')

if __name__ == '__main__':
    main()
