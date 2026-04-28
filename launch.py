"""
Detached process launcher for Windows.
Starts run_sp500.py then run_crypto.py as truly independent processes
that survive after this script exits.

Usage:
    python launch.py sp500       # run SP500 (deepseek pass)
    python launch.py crypto      # run Crypto (full pipeline)
    python launch.py sp500 crypto  # run both sequentially in background
"""

import sys, os, subprocess
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = sys.executable

SCRIPTS = {
    'sp500':  (ROOT / 'run_sp500.py',  ROOT / 'results' / 'sp500'  / 'SP500.log'),
    'crypto': (ROOT / 'run_crypto.py', ROOT / 'results' / 'crypto' / 'Crypto.log'),
    'vn30':   (ROOT / 'run_vn30.py',   ROOT / 'results' / 'vn30'   / 'VN30.log'),
}

DETACHED_PROCESS    = 0x00000008
CREATE_NEW_CONSOLE  = 0x00000010


def launch(name: str):
    script, log = SCRIPTS[name]
    log.parent.mkdir(parents=True, exist_ok=True)
    # Scripts write their own log via Python logging (SP500.log etc.).
    # stdout/stderr go to NUL; the real log is written by the script itself.
    devnull = open(os.devnull, 'w')
    flags = DETACHED_PROCESS if sys.platform == 'win32' else 0
    proc = subprocess.Popen(
        [PYTHON, str(script)],
        cwd=str(ROOT),
        stdin=devnull,
        stdout=devnull,
        stderr=devnull,
        creationflags=flags,
    )
    print(f'  Launched {name} (PID {proc.pid}) -> {log.relative_to(ROOT)}')
    return proc


if __name__ == '__main__':
    targets = sys.argv[1:] or ['sp500', 'crypto']
    invalid = [t for t in targets if t not in SCRIPTS]
    if invalid:
        print(f'Unknown targets: {invalid}. Choose from: {list(SCRIPTS)}')
        sys.exit(1)

    print(f'Launching: {targets}')
    procs = []
    for t in targets:
        p = launch(t)
        procs.append((t, p))

    print(f'\nAll launched. Monitor with:  python monitor.py')
    print(f'Or tail logs directly:')
    for t, p in procs:
        _, log = SCRIPTS[t]
        print(f'  {t}: {log}')
