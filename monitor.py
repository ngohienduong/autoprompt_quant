"""
AutoPrompt-Quant — live pipeline monitor.

Run in any terminal:
    python monitor.py

Press Ctrl+C to exit. Refreshes every 3 seconds.
Requires a terminal that supports ANSI escape codes
(Windows Terminal, VS Code integrated terminal, PowerShell 7).
"""

import io, os, re, sys, time, requests
from pathlib import Path
from datetime import datetime

# Force UTF-8 so box-drawing / block characters render on Windows Terminal
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = Path(__file__).parent
OLLAMA_BASE = 'http://localhost:11434'
REFRESH = 3  # seconds

# ── ANSI helpers ───────────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
BLUE   = '\033[94m'
WHITE  = '\033[97m'
GRAY   = '\033[90m'

def clr(text, *codes): return ''.join(codes) + str(text) + RESET
def clear_screen():    sys.stdout.write('\033[2J\033[H'); sys.stdout.flush()
def hide_cursor():     sys.stdout.write('\033[?25l'); sys.stdout.flush()
def show_cursor():     sys.stdout.write('\033[?25h'); sys.stdout.flush()

# ── Terminal width ─────────────────────────────────────────────────────────
def term_width():
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80

# ── Ollama ─────────────────────────────────────────────────────────────────
EXPECTED_MODELS = {
    'deepseek': 'deepseek-coder-v2:16b',
    'qwen':     'qwen2.5-coder:14b',
    'gemma':    'gemma2:9b',
    'llama':    'llama3.1:8b',
    'mistral':  'mistral:7b',
}

def get_ollama_models():
    try:
        r = requests.get(f'{OLLAMA_BASE}/api/tags', timeout=3)
        return {m['name']: m.get('size', 0) for m in r.json().get('models', [])}
    except Exception:
        return {}

# ── Log parsing ────────────────────────────────────────────────────────────
STAGE_RE    = re.compile(r'STAGE (\S+):')
MODEL_RE    = re.compile(r'\[(\w+)\]\s+(?:Generating|Coding|Loaded).*?(\d+)/(\d+)')
PROMPT_RE   = re.compile(r'\[(\w+)\]\s+(\d+)/(\d+)\s+arch=')
STRAT_RE    = re.compile(r'\[(\w+)\]\s+(\d+)/(\d+)\s+IS-Sharpe=')
WF_RE       = re.compile(r'WF progress:\s+(\d+)/(\d+)')
DONE_RE     = re.compile(r'experiment complete')
SKIP_RE     = re.compile(r'\[(\w+)\] SKIP')
DONE_ALT_RE = re.compile(r'\[(\w+)\] (?:Done|Loaded) (\d+)(?:/(\d+))? (?:prompts|strategies)')

def parse_log(path: Path):
    """Return a status dict from a market log file."""
    status = {
        'stage':    None,
        'model':    None,
        'progress': None,    # (current, total)
        'complete': False,
        'running':  False,
        'skipped':  [],
        'done_models': {},   # alias -> count
        'last_lines': [],
    }
    if not path.exists():
        return status

    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    status['last_lines'] = [l for l in lines if 'INFO' in l or 'WARNING' in l or 'ERROR' in l][-8:]

    # Walk backwards for freshest info
    for line in reversed(lines):
        if not line.strip():
            continue
        if DONE_RE.search(line) and status['complete'] is False:
            # Count completions (each full run = one completion)
            pass
        if status['stage'] is None:
            m = STAGE_RE.search(line)
            if m:
                status['stage'] = m.group(1)

        if status['model'] is None or status['progress'] is None:
            # WF progress
            m = WF_RE.search(line)
            if m:
                status['model'] = 'walkforward'
                status['progress'] = (int(m.group(1)), int(m.group(2)))
                continue
            # Strat coding progress
            m = STRAT_RE.search(line)
            if m and status['progress'] is None:
                status['model'] = m.group(1)
                status['progress'] = (int(m.group(2)), int(m.group(3)))
                continue
            # Prompt generation progress
            m = PROMPT_RE.search(line)
            if m and status['progress'] is None:
                status['model'] = m.group(1)
                status['progress'] = (int(m.group(2)), int(m.group(3)))
                continue

    # Count complete markers
    full_text = '\n'.join(lines)
    status['complete'] = full_text.count('experiment complete') > 0
    status['running']  = path.exists() and not status['complete']

    # Which models are skipped
    for line in lines:
        m = SKIP_RE.search(line)
        if m:
            alias = m.group(1)
            if alias not in status['skipped']:
                status['skipped'].append(alias)

    # Which models are fully done (strats checkpoint logged)
    for line in lines:
        m = re.search(r'\[(\w+)\] Done: (\d+)/\d+ (?:valid )?strategies', line)
        if m:
            status['done_models'][m.group(1)] = int(m.group(2))

    return status

ALIASES = list(EXPECTED_MODELS.keys())  # ['deepseek','qwen','gemma','llama','mistral']

def count_completions(path: Path) -> int:
    if not path.exists():
        return 0
    return path.read_text(encoding='utf-8', errors='ignore').count('experiment complete')

def strats_done(market: str) -> list:
    """Return list of model aliases that have a non-empty strats checkpoint."""
    ckpt = ROOT / 'results' / market.lower() / 'checkpoints'
    done = []
    for alias in ALIASES:
        f = ckpt / f'strats_{alias}.pkl'
        if f.exists() and f.stat().st_size > 100:
            done.append(alias)
    return done

def is_log_active(path: Path, window_sec: int = 180) -> bool:
    """True if log was written to within the last window_sec seconds.
    180s default: some LLM calls take 60-90s so we allow extra headroom."""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < window_sec

# ── Progress bar ───────────────────────────────────────────────────────────
def pbar(current, total, width=20, fill='█', empty='░'):
    if total == 0:
        frac = 0
    else:
        frac = min(current / total, 1.0)
    filled = int(frac * width)
    bar = fill * filled + empty * (width - filled)
    pct = int(frac * 100)
    return bar, pct

# ── Render ─────────────────────────────────────────────────────────────────
def render():
    W = min(term_width(), 100)
    now = datetime.now().strftime('%H:%M:%S')

    lines = []
    def line(s=''): lines.append(s)

    # ── Header ───────────────────────────────────────────────────────────
    title = 'AutoPrompt-Quant  Pipeline Monitor'
    pad   = (W - len(title) - 2) // 2
    line(clr('─' * W, CYAN))
    line(clr(' ' * pad + title + ' ' * pad, BOLD + CYAN))
    line(clr('─' * W, CYAN))
    line(clr(f'  Updated: {now}   Press Ctrl+C to exit', DIM))
    line()

    # ── Ollama models ─────────────────────────────────────────────────────
    available = get_ollama_models()
    line(clr('  OLLAMA MODELS', BOLD + WHITE))
    for alias, tag in EXPECTED_MODELS.items():
        if tag in available:
            sz = available[tag] / 1e9
            line(f'  {clr("✓", GREEN)} {tag:<30} {clr(f"{sz:.1f} GB", DIM)}')
        else:
            line(f'  {clr("○", GRAY)} {tag:<30} {clr("not available", GRAY)}')
    line()

    # ── Market pipeline status ─────────────────────────────────────────────
    line(clr('  EXPERIMENT PROGRESS', BOLD + WHITE))

    market_configs = [
        ('VN30',   ROOT / 'results' / 'vn30'   / 'VN30.log',     'VN30  '),
        ('SP500',  ROOT / 'results' / 'sp500'   / 'SP500.log',    'SP500 '),
        ('Crypto', ROOT / 'results' / 'crypto'  / 'Crypto.log',   'Crypto'),
    ]

    for market, log_path, label in market_configs:
        st       = parse_log(log_path)
        done_aliases = strats_done(market)
        n_done   = len(done_aliases)
        n_total  = len(ALIASES)
        active   = is_log_active(log_path)
        all_strats_complete = n_done == n_total

        if all_strats_complete and not active:
            color  = GREEN
            status = 'COMPLETE'
            bar, _ = pbar(n_done, n_total, 24)
        elif active or (log_path.exists() and not all_strats_complete):
            color  = YELLOW
            status = 'RUNNING ' if active else 'PAUSED  '
            if st['progress']:
                bar, _ = pbar(st['progress'][0], st['progress'][1], 24)
            else:
                bar, _ = pbar(n_done, n_total, 24)
        else:
            color  = GRAY
            status = 'WAITING '
            bar, _ = pbar(0, 1, 24)

        # Detail line
        model_summary = ''
        if done_aliases:
            model_summary = f'  [{"/".join(done_aliases)} ✓]'
        detail = ''
        if st['stage'] and active:
            detail += f'Stage {st["stage"]}'
        if st['model'] and st['progress'] and active:
            cur, tot = st['progress']
            detail += f'  [{st["model"]}] {cur}/{tot}'
        if not detail and model_summary:
            detail = f'models done:{model_summary}'

        bar_str = clr(bar, color)
        line(f'  {clr(label, BOLD)}  [{bar_str}] {clr(status, BOLD + color)}  {clr(detail, DIM)}')

    line()

    # ── run_remaining.py status ────────────────────────────────────────────
    rem_log = ROOT / 'results' / 'run_remaining.log'
    if rem_log.exists():
        rem_lines = rem_log.read_text(encoding='utf-8', errors='ignore').splitlines()
        if rem_lines:
            last = rem_lines[-1].strip()
            line(clr('  AUTOMATION (run_remaining.py)', BOLD + WHITE))
            line(f'  {clr(last[:W-4], DIM)}')
            line()

    # ── Live log tail ──────────────────────────────────────────────────────
    # Show tail from whichever market log was most recently updated
    active_log = None
    active_market = None
    newest_mtime = 0
    for market, log_path, label in market_configs:
        if log_path.exists():
            mtime = log_path.stat().st_mtime
            if mtime > newest_mtime:
                newest_mtime = mtime
                active_log = log_path
                active_market = label.strip()

    if active_log:
        st = parse_log(active_log)
        log_lines = st['last_lines'][-7:]
        line(clr(f'  LOG TAIL ({active_market})', BOLD + WHITE))
        for ll in log_lines:
            # Strip timestamp prefix for compactness
            ll = re.sub(r'^\d{4}-\d{2}-\d{2} ', '', ll)
            level_color = YELLOW if 'WARNING' in ll else (RED if 'ERROR' in ll else DIM)
            line(f'  {clr(ll[:W-4], level_color)}')

    line()
    line(clr('─' * W, CYAN))
    return '\n'.join(lines)


# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    # Enable ANSI on Windows
    if sys.platform == 'win32':
        os.system('')

    hide_cursor()
    try:
        while True:
            output = render()
            clear_screen()
            print(output, flush=True)
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        clear_screen()
        show_cursor()
        print('Monitor stopped.')


if __name__ == '__main__':
    main()
