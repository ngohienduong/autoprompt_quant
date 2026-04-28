# AutoPrompt-Quant  
## Reproducible Research Repository

> **AutoPrompt-Quant: Generating Statistically Significant Quantitative Trading Strategies with Open-Source LLMs — Walk-Forward Evidence Across Three Markets**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![License](https://img.shields.io/badge/License-MIT-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![Status](https://img.shields.io/badge/Status-Research%20Replication-success)

</div>

---

## Overview

AutoPrompt-Quant is a fully reproducible experimental framework for evaluating whether open-source LLMs can autonomously generate quantitative trading strategies with statistically meaningful out-of-sample performance.

The repository contains:

- Full experimental pipeline
- Market data fetchers
- Walk-forward evaluation engine
- Strategy clustering & diversity filtering
- Reproduction scripts for paper tables and figures
- Checkpoint-safe experiment runners

All results in the paper are generated from actual experiment runs 

---

# Markets Evaluated

| Market | Description |
|---|---|
| **VN30** | Vietnamese equities (HOSE) |
| **S&P 500** | US large-cap equities |
| **Crypto-10** | Top cryptocurrencies by market capitalization |

---

# Models Evaluated

| Model | Type |
|---|---|
| `mistral:7b` | General-purpose LLM |
| `llama3.1:8b` | Meta Llama |
| `gemma2:9b` | Google Gemma |
| `qwen2.5-coder:14b` | Coding-specialized |
| `deepseek-coder-v2:16b` | Coding-specialized |

All models run locally using **Ollama** with GGUF quantization on a RTX 4090.

---

# Experimental Results

---

## VN30 (Vietnamese Equities)

| Model | Valid Strategies | Mean OOS Sharpe | DSR Pass |
|---|---:|---:|---:|
| deepseek-coder-v2:16b | 23 | -0.208 | 0 / 23 |
| qwen2.5-coder:14b | 26 | -0.181 | 0 / 26 |
| gemma2:9b | 16 | -0.168 | 0 / 16 |
| llama3.1:8b | 43 | -0.071 | 0 / 43 |
| mistral:7b | 24 | -0.395 | 0 / 24 |
| **Combined** | **132** | **-0.187** | **0 / 132** |

### Key observations

- IS–OOS Pearson correlation: **0.683**
- Top-5 IS-selected Sharpe: **0.312**
- Buy-and-hold Sharpe: **0.541**

---

## S&P 500 (US Large-Cap Equities)

| Model | Valid Strategies | Mean OOS Sharpe | DSR Pass |
|---|---:|---:|---:|
| deepseek-coder-v2:16b | 25 | 0.458 | 1 / 25 |
| qwen2.5-coder:14b | 27 | 0.480 | 1 / 27 |
| gemma2:9b | 11 | 0.236 | 0 / 11 |
| llama3.1:8b | 16 | 0.228 | 0 / 16 |
| mistral:7b | 9 | 0.812 | 1 / 9 |
| **Combined** | **88** | **0.432** | **3 / 88** |

### Key observations

- IS–OOS Pearson correlation: **0.958**
- Outlier-adjusted correlation: **≈ 0.50**
- Top-5 IS-selected Sharpe: **3.072**
- Excluding outliers: **≈ 0.81**
- Buy-and-hold Sharpe: **1.243**

---

## Crypto-10

| Model | Valid Strategies | Mean OOS Sharpe | DSR Pass |
|---|---:|---:|---:|
| deepseek-coder-v2:16b | 27 | 0.359 | 0 / 27 |
| qwen2.5-coder:14b | 10 | 0.367 | 0 / 10 |
| gemma2:9b | 7 | 0.836 | 0 / 7 |
| llama3.1:8b | 24 | 0.537 | 0 / 24 |
| mistral:7b | 6 | 0.579 | 0 / 6 |
| **Combined** | **74** | **0.481** | **0 / 74** |

### Key observations

- IS–OOS Pearson correlation: **0.825**
- Top-5 IS-selected Sharpe: **1.019**
- Buy-and-hold Sharpe: **0.612**

---

# System Requirements

## Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | CPU-only supported | RTX 4090 (24 GB VRAM) |
| RAM | 16 GB | 32 GB |
| Storage | 60 GB | 100 GB |

The paper experiments used a single RTX 4090 with 4-bit GGUF quantized models.

---

## Software

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| Ollama | Latest |
| CUDA | 12.x recommended |

---

# Installation

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/autoprompt-quant-experiments
cd autoprompt-quant-experiments
```

---

## 2. Create Virtual Environment

### Linux / macOS / WSL

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Required Python Packages

| Package | Purpose |
|---|---|
| `numpy`, `pandas`, `scipy` | Numerical computing |
| `scikit-learn`, `statsmodels` | Statistical analysis |
| `sentence-transformers` | Embedding generation |
| `hdbscan` | Clustering |
| `matplotlib` | Figure generation |
| `yfinance` | SP500 market data |
| `quantvn` | VN30 data |
| `requests` | API communication |

---

# Ollama Setup

AutoPrompt-Quant runs all LLMs locally through Ollama.

---

## Linux / macOS

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

---

## Windows

Download Ollama:

```text
https://ollama.com
```

---

## WSL2 (Recommended for Windows)

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
```

Ollama will be accessible at:

```text
localhost:11434
```

---

# Download Models

```bash
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b
ollama pull qwen2.5-coder:14b
ollama pull deepseek-coder-v2:16b
```

---

## Approximate VRAM Usage

| Model | VRAM |
|---|---|
| mistral:7b | ~4 GB |
| llama3.1:8b | ~5 GB |
| gemma2:9b | ~5 GB |
| qwen2.5-coder:14b | ~9 GB |
| deepseek-coder-v2:16b | ~9 GB |

Verify installation:

```bash
ollama list
```

---

# API Configuration

Copy template:

```bash
cp config.example.py config.py
```

Edit `config.py`:

```python
QUANTVN_API_KEY = "your_quantvn_key"
BINANCE_API_KEY = "your_binance_key"
```

Or use environment variables:

```bash
export QUANTVN_API_KEY="your_quantvn_key"
export BINANCE_API_KEY="your_binance_key"
```

---

# Data Sources

| Market | Source | API Key |
|---|---|---|
| VN30 | QuantVN API | Required |
| S&P 500 | Yahoo Finance | Not required |
| Crypto-10 | Binance REST API | Required |

A read-only Binance API key is sufficient.

---

# Fetch Market Data

---

## Fetch All Markets

```bash
python fetch_data.py
```

---

## Fetch Specific Markets

```bash
python fetch_data.py sp500
python fetch_data.py vn30 crypto
```

---

# Dataset Configuration

| Market | Date Range | IS Period | OOS Windows |
|---|---|---|---|
| VN30 | 2017–2024 | 2018–2020 | 8 × 6-month |
| S&P 500 | 2017–2024 | 2018–2020 | 8 × 6-month |
| Crypto-10 | 2019–2024 | 2020–2022 | 4 × 6-month |

An additional warm-up year is included for indicator initialization.

---

# Running Experiments

---

## Run All Markets

```bash
python run_all.py
```

---

## Run Individual Markets

```bash
python run_vn30.py
python run_sp500.py
python run_crypto.py
```

---

## Run a Subset

```bash
python run_all.py sp500 crypto
```

---

# Monitor Progress

```bash
python monitor.py
```

---

# Runtime Estimates

Measured on RTX 4090 with 50 prompts per model.

| Market | Runtime |
|---|---|
| VN30 | ~4 hours |
| S&P 500 | ~4 hours |
| Crypto-10 | ~3 hours |

CPU-only execution is approximately **10–30× slower**.

---

# Pipeline Overview

The framework consists of five stages:

| Stage | Description |
|---|---|
| **1. Meta-Agent** | Prompt generation via Latin-hypercube sampling |
| **2. Coding Agent** | Converts prompts into `gen_position()` code |
| **3. IS Optimization** | Parameter grid search |
| **4. Clustering** | Removes redundant strategies |
| **5. Walk-Forward OOS** | Rolling out-of-sample evaluation |

---

# Repository Structure

```text
autoprompt-quant-experiments/
├── fetch_data.py
├── run_all.py
├── run_vn30.py
├── run_sp500.py
├── run_crypto.py
├── monitor.py
├── launch.py
├── generate_paper_figures.py
├── stats_check.py
├── requirements.txt
├── config.example.py
├── src/
│   └── shared.py
├── data/
│   └── fetchers.py
└── results/
```

---

# Output Structure

```text
results/
├── vn30/
├── sp500/
└── crypto/
```

Each market directory contains:

```text
checkpoints/
├── prompts_{model}.pkl
├── strats_{model}.pkl
├── clustering.pkl
├── oos_results.pkl
├── oos_summary.csv
└── cross_model.csv
```

Large cached datasets and `.pkl` files are excluded from Git.

---

# Reproducing Figures

```bash
python generate_paper_figures.py
```

Generated figures are written to:

```text
../figures/
```

---

# Reproducing Paper Statistics

```bash
python stats_check.py
```

This reproduces:

- Pearson correlations
- OLS calibration
- DSR significance tests
- Top-5 Sharpe metrics
- Walk-forward statistics

---

# Citation

```bibtex
@article{autoprompt_quant_2025,
  title   = {AutoPrompt-Quant: Generating Statistically Significant Quantitative Trading Strategies with Open-Source LLMs --- Walk-Forward Evidence Across Three Markets},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025},
}
```

---

# License

MIT License — see `LICENSE` for details.

---


