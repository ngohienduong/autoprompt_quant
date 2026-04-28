"""
Run remaining experiments to completion and commit results to git.

Usage — run in a NEW terminal (SP500 may still be running in another):
    python run_remaining.py

What it does:
  1. Waits for the currently-running SP500 run to finish
     (polls sp500_run.log for the completion marker).
  2. Waits for deepseek-coder-v2:16b to finish downloading in Ollama.
  3. Re-runs SP500 (adds deepseek; all other models load from checkpoint).
  4. Runs Crypto (full pipeline, all 5 models).
  5. Commits updated CSVs + figures to git.
"""

import sys, time, subprocess, requests
from pathlib import Path

ROOT = Path(__file__).parent
OLLAMA_BASE  = 'http://localhost:11434'
DEEPSEEK_TAG = 'deepseek-coder-v2:16b'
SP500_LOG    = ROOT / 'results' / 'sp500' / 'sp500_run.log'
POLL_INTERVAL = 30   # seconds between checks


def ollama_list():
    try:
        r = requests.get(f'{OLLAMA_BASE}/api/tags', timeout=5)
        return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return []


def sp500_finished():
    if not SP500_LOG.exists():
        return False
    text = SP500_LOG.read_text(encoding='utf-8', errors='ignore')
    # Count how many times "experiment complete" appears;
    # each full run writes it once.
    completions = text.count('[SP500] experiment complete')
    # We need at least 2: the mistral-only run + the qwen/gemma/llama run
    return completions >= 2


def run_script(script: str, log: str):
    print(f'\n{"="*60}')
    print(f'  Running: {script}')
    print(f'  Log:     results/{Path(log).name}')
    print(f'{"="*60}')
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'a') as f:
        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            stdout=f, stderr=subprocess.STDOUT,
        )
    if result.returncode == 0:
        print(f'  Completed OK')
    else:
        print(f'  WARNING: exit code {result.returncode} — check {log}')
    return result.returncode


def commit_results():
    result_files = (
        list(ROOT.glob('results/**/*.csv'))
        + list(ROOT.glob('results/**/*.png'))
    )
    if not result_files:
        print('No result files to commit.')
        return
    for f in result_files:
        subprocess.run(['git', '-C', str(ROOT), 'add', str(f)], check=False)
    subprocess.run(
        ['git', '-C', str(ROOT), 'commit', '-m',
         'Add completed experiment results (all 5 models, all 3 markets)\n\n'
         'Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>'],
        check=False,
    )
    print('Results committed to git.')


# ── Step 1: Wait for SP500 second run to finish ────────────────────────────
if not sp500_finished():
    print('Waiting for current SP500 run (qwen/gemma/llama) to finish...')
    while not sp500_finished():
        print(f'  [{time.strftime("%H:%M:%S")}] SP500 still running — checking again in {POLL_INTERVAL}s')
        time.sleep(POLL_INTERVAL)
    print('  SP500 second run complete.')
else:
    print('SP500 second run already complete.')

# ── Step 2: Wait for deepseek ─────────────────────────────────────────────
if DEEPSEEK_TAG not in ollama_list():
    print(f'\nWaiting for {DEEPSEEK_TAG} to finish downloading...')
    while DEEPSEEK_TAG not in ollama_list():
        print(f'  [{time.strftime("%H:%M:%S")}] Not yet available — checking in {POLL_INTERVAL}s')
        time.sleep(POLL_INTERVAL)
print(f'  {DEEPSEEK_TAG} is ready.')

# ── Step 3: SP500 — add deepseek ──────────────────────────────────────────
print('\nRunning SP500 (final pass — deepseek only, others from checkpoint)...')
run_script('run_sp500.py', str(SP500_LOG))

# ── Step 4: Crypto — full pipeline ────────────────────────────────────────
print('\nRunning Crypto (full pipeline — all 5 models)...')
run_script('run_crypto.py', str(ROOT / 'results' / 'crypto' / 'crypto_run.log'))

# ── Step 5: Commit results ─────────────────────────────────────────────────
print('\nCommitting results to git...')
commit_results()

print('\nAll experiments complete.')
print('Push to GitHub:  bash push_to_github.sh')
print('Or via Windows:  git push -u origin main')
