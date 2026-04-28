"""
Generate final paper figures from completed experiment results.

Reads:
  results/{vn30,sp500,crypto}/checkpoints/oos_results.pkl
  results/{vn30,sp500,crypto}/checkpoints/cross_model.csv

Writes to:
  ../figures/   (the paper's figure directory)
  results/{market}/figures/  (per-market copies)
"""

import pickle, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT     = Path(__file__).parent
FIG_DIR  = ROOT / '..' / 'figures'          # paper's figure folder
FIG_DIR.mkdir(exist_ok=True)

MARKETS = {
    'VN30':   ('vn30',   '#1f77b4', 252,
               ['2021H1','2021H2','2022H1','2022H2','2023H1','2023H2','2024H1','2024H2']),
    'SP500':  ('sp500',  '#2ca02c', 252,
               ['2021H1','2021H2','2022H1','2022H2','2023H1','2023H2','2024H1','2024H2']),
    'Crypto': ('crypto', '#d62728', 365,
               ['2023H1','2023H2','2024H1','2024H2']),
}
MODEL_COLORS = {
    'deepseek': '#1f77b4', 'qwen': '#ff7f0e',
    'gemma':    '#2ca02c', 'llama': '#d62728', 'mistral': '#9467bd',
}
MODEL_MARKERS = {'deepseek': 'o', 'qwen': 's', 'gemma': '^', 'llama': 'D', 'mistral': 'P'}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'figure.dpi': 150,
})


def load(market_key):
    folder, _, _, _ = MARKETS[market_key]
    ckpt = ROOT / 'results' / folder / 'checkpoints'
    oos  = pd.read_pickle(ckpt / 'oos_results.pkl')
    cross = pd.read_csv(ckpt / 'cross_model.csv')
    return oos, cross


# ── Fig 1: IS vs OOS scatter — all three markets (3-panel row) ────────────
def fig_is_oos_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('IS vs OOS Sharpe Ratio by Model and Market', fontsize=12, fontweight='bold')

    for ax, (mkt, (folder, mcolor, ann, _)) in zip(axes, MARKETS.items()):
        oos, _ = load(mkt)
        df_agg = (oos.groupby(['alias','prompt_hash'])['oos_sharpe']
                  .mean().reset_index()
                  .rename(columns={'oos_sharpe': 'mean_oos_sharpe'}))
        strat_pkl = list((ROOT / 'results' / folder / 'checkpoints').glob('strats_*.pkl'))
        is_map = {}
        for p in strat_pkl:
            with open(p, 'rb') as f:
                strats = pickle.load(f)
            import hashlib
            for s in strats:
                h = hashlib.md5(s['prompt'].encode()).hexdigest()[:8]
                is_map[h] = s['is_sharpe']
        df_agg['is_sharpe'] = df_agg['prompt_hash'].map(is_map)
        mask = df_agg['is_sharpe'].notna() & df_agg['mean_oos_sharpe'].notna()

        for alias in sorted(df_agg['alias'].unique()):
            sub = df_agg[df_agg['alias'] == alias]
            ax.scatter(sub['is_sharpe'], sub['mean_oos_sharpe'],
                       color=MODEL_COLORS.get(alias,'gray'),
                       marker=MODEL_MARKERS.get(alias,'o'),
                       alpha=0.75, s=45, label=alias, zorder=3)

        if mask.sum() > 5:
            xs = df_agg.loc[mask, 'is_sharpe']
            ys = df_agg.loc[mask, 'mean_oos_sharpe']
            m, b = np.polyfit(xs, ys, 1)
            xr = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xr, m*xr+b, 'k--', lw=1.2, alpha=0.6, label='OLS fit')
            r = np.corrcoef(xs, ys)[0,1]
            n = mask.sum()
            ax.set_title(f'{mkt}  (r={r:.2f}, N={n})')
        else:
            ax.set_title(mkt)

        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.axvline(0, color='gray', lw=0.7, ls='--')
        ax.set_xlabel('IS Sharpe Ratio')
        ax.set_ylabel('Mean OOS Sharpe Ratio')
        ax.legend(ncol=2, framealpha=0.8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig5_is_oos_scatter.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(ROOT / 'results' / 'fig_is_oos_scatter_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Fig 2: Walk-forward OOS by market and model (3-row panel) ─────────────
def fig_walkforward():
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.suptitle('Walk-Forward Out-of-Sample Sharpe Ratio', fontsize=12, fontweight='bold')

    for ax, (mkt, (folder, mcolor, ann, xlabels)) in zip(axes, MARKETS.items()):
        oos, _ = load(mkt)
        wf_mean  = oos.groupby(['alias','window'])['oos_sharpe'].mean().reset_index()
        cross_m  = oos.groupby('window')['oos_sharpe'].mean()

        ax.plot(cross_m.index, cross_m.values, 'k-', lw=2.5,
                label='Cross-model mean', zorder=5)
        ax.fill_between(cross_m.index,
                         oos.groupby('window')['oos_sharpe'].quantile(0.25),
                         oos.groupby('window')['oos_sharpe'].quantile(0.75),
                         alpha=0.12, color='gray', label='IQR')

        for alias in sorted(oos['alias'].unique()):
            sub = wf_mean[wf_mean['alias'] == alias]
            ax.plot(sub['window'], sub['oos_sharpe'],
                    color=MODEL_COLORS.get(alias,'gray'),
                    marker=MODEL_MARKERS.get(alias,'o'),
                    ms=5, lw=1.2, alpha=0.8, label=alias)

        ax.axhline(0, color='red', lw=0.8, ls='--', alpha=0.7)
        ax.set_title(mkt, fontweight='bold')
        ax.set_ylabel('Mean OOS Sharpe')
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=20)
        ax.legend(ncol=4, framealpha=0.8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig4_walkforward.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Fig 3: Cross-model comparison table as heatmap ────────────────────────
def fig_cross_model():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Cross-Model Strategy Performance Summary', fontsize=12, fontweight='bold')

    metrics = ['mean_oos', 'dsr_pass_rate', 'nondegen']
    metric_labels = ['Mean OOS Sharpe', 'DSR Pass Rate', 'Non-Degenerate Rate']
    cmaps = ['RdYlGn', 'YlOrRd', 'Blues']

    for ax, metric, label, cmap in zip(axes, metrics, metric_labels, cmaps):
        rows = []
        for mkt, (folder, *_) in MARKETS.items():
            cross = pd.read_csv(ROOT / 'results' / folder / 'checkpoints' / 'cross_model.csv')
            cross['market'] = mkt
            rows.append(cross[['alias','market', metric]])
        df = pd.concat(rows).pivot(index='market', columns='alias', values=metric)
        df = df[['deepseek','qwen','gemma','llama','mistral']]
        df.index = list(MARKETS.keys())

        im = ax.imshow(df.values.astype(float), cmap=cmap, aspect='auto',
                       vmin=df.values.astype(float).min(),
                       vmax=df.values.astype(float).max())
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                v = df.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, color='black')

    plt.tight_layout()
    out = FIG_DIR / 'fig6_cross_asset.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Fig 4: OOS Sharpe distribution boxplot — all markets ─────────────────
def fig_oos_boxplot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('OOS Sharpe Distribution by Model', fontsize=12, fontweight='bold')

    for ax, (mkt, (folder, mcolor, ann, _)) in zip(axes, MARKETS.items()):
        oos, _ = load(mkt)
        df_agg = (oos.groupby(['alias','prompt_hash'])['oos_sharpe']
                  .mean().reset_index())
        aliases = sorted(df_agg['alias'].unique())
        data    = [df_agg[df_agg['alias']==a]['oos_sharpe'].dropna().values for a in aliases]
        bp = ax.boxplot(data, labels=aliases, patch_artist=True, notch=False,
                        medianprops=dict(color='black', lw=2))
        for patch, alias in zip(bp['boxes'], aliases):
            patch.set_facecolor(MODEL_COLORS.get(alias, 'lightblue'))
            patch.set_alpha(0.8)
        ax.axhline(0, color='red', lw=0.9, ls='--', label='Zero')
        ax.set_title(mkt, fontweight='bold')
        ax.set_ylabel('Mean OOS Sharpe Ratio')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, axis='y', alpha=0.3)
        n_strats = {a: (df_agg['alias']==a).sum() for a in aliases}
        for i, alias in enumerate(aliases):
            ax.text(i+1, ax.get_ylim()[0]*0.95, f'n={n_strats[alias]}',
                    ha='center', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()
    out = FIG_DIR / 'fig3_dimension_sharpe.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Fig 5: DSR pass rate bar chart ────────────────────────────────────────
def fig_dsr():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Deflated Sharpe Ratio (DSR) Pass Rate by Model (α=0.05)',
                 fontsize=12, fontweight='bold')

    for ax, (mkt, (folder, mcolor, ann, _)) in zip(axes, MARKETS.items()):
        _, cross = load(mkt)
        dsr = cross.set_index('alias')['dsr_pass_rate'] * 100
        dsr = dsr.reindex(['deepseek','qwen','gemma','llama','mistral']).fillna(0)
        bars = ax.bar(dsr.index,
                      dsr.values,
                      color=[MODEL_COLORS.get(a,'steelblue') for a in dsr.index],
                      edgecolor='k', alpha=0.85)
        ax.axhline(5, color='red', lw=1.2, ls='--', label='5% threshold')
        ax.set_ylim(0, max(dsr.values.max() * 1.3, 15))
        ax.set_ylabel('DSR Pass Rate (%)')
        ax.set_title(mkt, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, dsr.values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / 'main_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Fig 6: cumulative OOS Sharpe trajectory (simulated equity index) ──────
def fig_cumulative():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Cumulative Mean OOS Sharpe — Walk-Forward Windows',
                 fontsize=12, fontweight='bold')

    for ax, (mkt, (folder, mcolor, ann, xlabels)) in zip(axes, MARKETS.items()):
        oos, _ = load(mkt)
        wf_mean  = oos.groupby(['alias','window'])['oos_sharpe'].mean().reset_index()
        cross_m  = oos.groupby('window')['oos_sharpe'].mean()
        cum_cross = cross_m.cumsum()

        ax.plot(cum_cross.index, cum_cross.values, 'k-', lw=2.5,
                label='Cross-model cumulative', zorder=5)
        for alias in sorted(oos['alias'].unique()):
            sub = wf_mean[wf_mean['alias']==alias].sort_values('window')
            cum = sub['oos_sharpe'].cumsum()
            ax.plot(sub['window'], cum.values,
                    color=MODEL_COLORS.get(alias,'gray'),
                    marker=MODEL_MARKERS.get(alias,'o'),
                    ms=4, lw=1.2, alpha=0.75, label=alias)

        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_title(mkt, fontweight='bold')
        ax.set_ylabel('Cumulative OOS Sharpe')
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=20)
        ax.legend(ncol=2, framealpha=0.8, fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig1_cumulative_returns.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    print('Generating paper figures from experiment results...')
    fig_is_oos_scatter()
    fig_walkforward()
    fig_cross_model()
    fig_oos_boxplot()
    fig_dsr()
    fig_cumulative()
    print(f'\nAll figures saved to: {FIG_DIR.resolve()}')
    print('Paper can now be compiled with: pdflatex main.tex')
