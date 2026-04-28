# AutoPrompt-Quant — Reproducible Experiments

Replication package for the paper:

> **"AutoPrompt-Quant: Generating Statistically Significant Quantitative Trading Strategies with Open-Source LLMs — Walk-Forward Evidence Across Three Markets"**

All results below are from the actual experiment runs (not synthetic data).

---

## Results summary

### VN30 (Vietnamese equities, HOSE)

| Model | Valid strategies | Mean OOS Sharpe | DSR pass |
|-------|-----------------|----------------|---------|
| deepseek-coder-v2:16b | 23 | −0.208 | 0 / 23 |
| qwen2.5-coder:14b | 26 | −0.181 | 0 / 26 |
| gemma2:9b | 16 | −0.168 | 0 / 16 |
| llama3.1:8b | 43 | −0.071 | 0 / 43 |
| mistral:7b | 24 | −0.395 | 0 / 24 |
| **All combined** | **132** | **−0.187** | **0 / 132** |

IS-OOS Pearson r = 0.683 (p < 0.001). Top-5 IS-screened SR = 0.312. B&H SR = 0.541.

### SP500 (US large-cap equities, top-30)

| Model | Valid strategies | Mean OOS Sharpe | DSR pass |
|-------|-----------------|----------------|---------|
| deepseek-coder-v2:16b | 25 | 0.458 | 1 / 25 |
| qwen2.5-coder:14b | 27 | 0.480 | 1 / 27 |
| gemma2:9b | 11 | 0.236 | 0 / 11 |
| llama3.1:8b | 16 | 0.228 | 0 / 16 |
| mistral:7b | 9 | 0.812 | 1 / 9 |
| **All combined** | **88** | **+0.432** | **3 / 88** |

IS-OOS Pearson r = 0.958 (p < 0.001; outlier-adjusted r ≈ 0.50). Top-5 IS-screened SR = 3.072†. B&H SR = 1.243.
†Dominated by two outlier strategies; excluding them gives top-5 SR ≈ 0.81.

### Crypto-10 (top-10 by market cap)

| Model | Valid strategies | Mean OOS Sharpe | DSR pass |
|-------|-----------------|----------------|---------|
| deepseek-coder-v2:16b | 27 | 0.359 | 0 / 27 |
| qwen2.5-coder:14b | 10 | 0.367 | 0 / 10 |
| gemma2:9b | 7 | 0.836 | 0 / 7 |
| llama3.1:8b | 24 | 0.537 | 0 / 24 |
| mistral:7b | 6 | 0.579 | 0 / 6 |
| **All combined** | **74** | **+0.481** | **0 / 74** |

IS-OOS Pearson r = 0.825 (p < 0.001). Top-5 IS-screened SR = 1.019. B&H SR = 0.612.

---

## Requirements

### Hardware

| Component | Minimum | Recommended (paper setup) |
|-----------|---------|--------------------------|
| GPU VRAM | — (CPU works, ~10–30× slower per call) | 24 GB (RTX 4090) |
| RAM | 16 GB | 32 GB |
| Disk | 60 GB free | 100 GB |

The paper used a single RTX 4090 (24 GB VRAM). All five models fit simultaneously under 11 GB with 4-bit GGUF quantisation.

### Software

| Component | Version used | Notes |
|-----------|-------------|-------|
| Python | 3.12 | 3.10+ supported |
| Ollama | latest | serves LLMs locally |
| CUDA | 12.x | for GPU inference; CPU works without it |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/autoprompt-quant-experiments
cd autoprompt-quant-experiments
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv

# Activate (Linux / macOS / WSL)
source .venv/bin/activate

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas`, `scipy` | numerical computing |
| `scikit-learn`, `statsmodels` | OLS regression, metrics |
| `hdbscan` | strategy clustering (Stage 4) |
| `sentence-transformers` | prompt diversity filtering (Stage 4) |
| `yfinance` | SP500 data (free, no key needed) |
| `quantvn` | VN30 data via QuantVN API |
| `requests` | Binance REST API (Crypto) + Ollama API |
| `matplotlib` | figure generation |

### 4. Install and configure Ollama

Ollama serves all five LLMs locally.

**Standard (Linux / macOS):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve   # leave running in a separate terminal
```

**Windows:**

Download and install from [https://ollama.com](https://ollama.com).
Ollama starts automatically as a background service.

**WSL2 on Windows (recommended if native GPU detection fails):**

```bash
# Inside WSL2:
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama   # or: ollama serve &
```

WSL2 auto-forwards port 11434 — Python on Windows reaches it at `localhost:11434`.

### 5. Pull the five LLM models (~33 GB total, one-time download)

```bash
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b
ollama pull qwen2.5-coder:14b
ollama pull deepseek-coder-v2:16b
```

Approximate sizes:

| Model | Tag | VRAM (Q4_K_M) |
|-------|-----|--------------|
| Mistral | `mistral:7b` | ~4 GB |
| LLaMA 3.1 | `llama3.1:8b` | ~5 GB |
| Gemma 2 | `gemma2:9b` | ~5 GB |
| Qwen 2.5 Coder | `qwen2.5-coder:14b` | ~9 GB |
| DeepSeek Coder V2 | `deepseek-coder-v2:16b` | ~9 GB |

Verify all models are available:

```bash
ollama list
```

### 6. Configure API keys

```bash
cp config.example.py config.py
```

Edit `config.py` and fill in your keys:

```python
QUANTVN_API_KEY = "your-quantvn-key-here"   # required for VN30
BINANCE_API_KEY = "your-binance-key-here"    # required for Crypto
```

Keys can also be set as environment variables instead:

```bash
export QUANTVN_API_KEY="your-quantvn-key-here"
export BINANCE_API_KEY="your-binance-key-here"
```

`config.py` is listed in `.gitignore` and will never be committed.

| Market | Data source | API key required |
|--------|-------------|-----------------|
| VN30 | [QuantVN API](https://quantvn.com/) | Yes — `QUANTVN_API_KEY` |
| SP500 | [yfinance](https://pypi.org/project/yfinance/) | No |
| Crypto-10 | [Binance REST API](https://www.binance.com/en/binance-api) | Yes — read-only `BINANCE_API_KEY` |

A **read-only** Binance key (no trading permissions) is sufficient for data fetching.

---

## Fetching data

Download and cache all market data before running experiments.  This only
needs to be done once — subsequent runs load from the local cache.

```bash
python fetch_data.py            # all three markets
python fetch_data.py sp500      # SP500 only (no API key needed)
python fetch_data.py vn30 crypto
```

Expected output (first run):

```
[SP500] Downloading 30 symbols via yfinance...
  AAPL: 2013 rows (2017-01-03 -> 2024-12-31)
  ...
[SP500] Saved 30 symbols to results/sp500/data/sp500_data.pkl

[Crypto] Fetching 10 symbols from Binance REST API...
  BTC (BTCUSDT): 2191 rows (2019-01-01 -> 2024-12-31)
  ...
[Crypto] Saved 10 symbols to results/crypto/data/crypto_data.pkl

[VN30] QuantVN authenticated.
[VN30] Fetching 30 symbols via QuantVN...
  ACB: 1823 rows (2017-01-03 -> 2024-12-31)
  ...
[VN30] Saved 30 symbols to results/vn30/checkpoints/price_data.pkl
```

Data spans (matching paper Section IV):

| Market | Symbols | Fetch range | IS period | OOS windows |
|--------|---------|------------|-----------|------------|
| VN30 | 30 HOSE constituents | 2017-01-01 → 2024-12-31 | 2018–2020 | 8 × 6-month (2021–2024) |
| SP500 | Top 30 by mkt cap | 2017-01-01 → 2024-12-31 | 2018–2020 | 8 × 6-month (2021–2024) |
| Crypto-10 | BTC ETH BNB XRP ADA SOL DOT AVAX LINK DOGE | 2019-01-01 → 2024-12-31 | 2020–2022 | 4 × 6-month (2023–2024) |

One extra year before the IS start date is included as indicator warm-up.

---

## Running experiments

After data is cached, run one market or all three:

```bash
python run_all.py               # all three markets sequentially
python run_vn30.py              # VN30 only
python run_sp500.py             # SP500 only
python run_crypto.py            # Crypto only
python run_all.py sp500 crypto  # subset
```

Monitor live progress in a second terminal:

```bash
python monitor.py
```

Each run is **checkpoint-safe** — interrupt and restart at any time without
losing completed work.  Each pipeline stage writes its output to
`results/<market>/checkpoints/` before the next stage begins.

Estimated runtimes on RTX 4090 (all 5 models, 50 prompts each):

| Market | ~Runtime |
|--------|---------|
| VN30 | 4 h |
| SP500 | 4 h |
| Crypto-10 | 3 h |

CPU-only is approximately 10–30× slower per LLM call.

---

## Reproducing paper figures

```bash
python generate_paper_figures.py
```

Reads `results/*/checkpoints/oos_results.pkl` and writes figures directly
to the paper's `../figures/` directory so that `pdflatex main.tex` picks
them up without copying.

---

## Pipeline overview

Five sequential stages (paper Section III):

| Stage | Description | Key parameters |
|-------|-------------|----------------|
| **1 — Meta-agent** | 50 prompts/model via Latin-hypercube sampling over a 4D strategy taxonomy (81 archetypes); cosine-similarity diversity filter τ = 0.85 | `N_PROMPTS=50`, `DIVERSITY_TAU=0.85` |
| **2 — Coding agent** | Converts each prompt to `gen_position()` Python code; escalating temperatures [0.2, 0.5, 0.8] on retry | `CODE_TEMPS=[0.2,0.5,0.8]` |
| **3 — IS grid search** | Grid: `lookback ∈ {5,10,20}` × `threshold ∈ {0,1,2}%`; best IS Sharpe across all symbols | `PARAM_GRID` in `src/shared.py` |
| **4 — Clustering** | HDBSCAN on sentence-transformer embeddings (L1); k-means on IS performance features, k by silhouette score (L2) | `hdbscan`, `sentence-transformers` |
| **5 — Walk-forward OOS** | Rolling 6-month windows; DSR test at α = 0.05; IS-OOS OLS with HC3 standard errors | `WF_WINDOWS` in each runner |

---

## Project structure

```
experiments_reproducible/
├── fetch_data.py               # one-shot data download (run first)
├── run_all.py                  # master runner — all three markets
├── run_vn30.py                 # VN30 experiment
├── run_sp500.py                # SP500 experiment
├── run_crypto.py               # Crypto experiment
├── launch.py                   # detached background process launcher
├── monitor.py                  # live terminal dashboard
├── generate_paper_figures.py   # regenerate figures from results
├── stats_check.py              # reproduce paper statistics from CSVs
├── config.example.py           # API key template
├── config.py                   # your keys — gitignored
├── requirements.txt
├── src/
│   └── shared.py               # all five pipeline stages (market-agnostic)
└── data/
    └── fetchers.py             # fetch_vn30 / fetch_sp500 / fetch_crypto
```

---

## Output layout

```
results/
├── vn30/
│   ├── checkpoints/
│   │   ├── price_data.pkl          # raw OHLCV cache (gitignored)
│   │   ├── prompts_{model}.pkl     # Stage 1 output
│   │   ├── strats_{model}.pkl      # Stage 2+3 output
│   │   ├── clustering.pkl          # Stage 4 output
│   │   ├── oos_results.pkl         # Stage 5 walk-forward rows
│   │   ├── oos_summary.csv         # human-readable summary (committed)
│   │   └── cross_model.csv         # cross-model comparison (committed)
│   └── figures/
│       ├── fig1_results.png
│       └── fig2_walkforward.png
├── sp500/  (same structure, data/ subdirectory instead of checkpoints/)
└── crypto/ (same structure)
```

`.pkl` files and `data/` directories are excluded from git (too large for version control).
CSV and PNG outputs are committed.

---

## Reproducing paper statistics

```bash
python stats_check.py
```

Reads the committed CSVs (`oos_summary.csv`, `cross_model.csv`) and
prints the IS-OOS Pearson r, OLS calibration, DSR counts, top-5 SR,
and window-level Sharpe for all three markets — matching Tables II–IV
in the paper.

Requires `scipy`:

```bash
pip install scipy
```

---

## Citation

```bibtex
@article{autoprompt_quant_2025,
  title   = {AutoPrompt-Quant: Generating Statistically Significant Quantitative
             Trading Strategies with Open-Source LLMs---Walk-Forward Evidence
             Across Three Markets},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025},
}
```

---

## License

MIT License — see `LICENSE` for details.
