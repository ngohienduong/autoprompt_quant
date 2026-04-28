import pandas as pd, numpy as np
from scipy import stats

markets = [
    ('vn30',   'results/vn30/checkpoints/oos_summary.csv',   'results/vn30/checkpoints/cross_model.csv'),
    ('sp500',  'results/sp500/checkpoints/oos_summary.csv',  'results/sp500/checkpoints/cross_model.csv'),
    ('crypto', 'results/crypto/checkpoints/oos_summary.csv', 'results/crypto/checkpoints/cross_model.csv'),
]
order = ['deepseek','qwen','gemma','llama','mistral']

for mkt, oos_f, cross_f in markets:
    print(f'\n=== {mkt.upper()} ===')
    oos   = pd.read_csv(oos_f)
    cross = pd.read_csv(cross_f)

    sm = oos.groupby(['alias','prompt_hash'])['oos_sharpe'].mean()
    si = oos.groupby(['alias','prompt_hash'])['is_sharpe'].first()
    n_win = oos['window'].nunique()
    print(f'windows={n_win}')

    for a in order:
        lvl0 = sm.index.get_level_values(0)
        if a not in lvl0:
            continue
        m = sm[a]; i = si[a]
        r, p = stats.pearsonr(i.values, m.values) if len(m) > 2 else (float('nan'), float('nan'))
        print(f'  {a}: n={len(m)}, OOS={m.mean():.3f}+/-{m.std():.3f}, IS={i.mean():.3f}+/-{i.std():.3f}, r={r:.3f}(p={p:.4f})')

    ai = si.values; ao = sm.values
    r2, p2 = stats.pearsonr(ai, ao)
    sl, ic, rv, pv, se = stats.linregress(ai, ao)
    n = len(ai); fscore = rv**2 / ((1-rv**2)/(n-2))
    print(f'  ALL: N={n}, r={r2:.3f}(p={p2:.5f}), OLS ic={ic:.3f}, sl={sl:.3f}, R2={rv**2:.3f}, F={fscore:.1f}')
    print(f'  OOS mean={ao.mean():.3f}+/-{ao.std():.3f}, IS mean={ai.mean():.3f}+/-{ai.std():.3f}')

    wf = oos.groupby('window')['oos_sharpe'].mean()
    print(f'  Window means: {[round(v,3) for v in wf.values]}')
    print(f'  max={wf.max():.3f}(w{wf.idxmax()}), min={wf.min():.3f}(w{wf.idxmin()}), swing={wf.max()-wf.min():.3f}')
