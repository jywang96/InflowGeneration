# %% [markdown]
# # Experiment 2 -- POD-GPR vs the shipped point-wise GPR
#
# Head-to-head accuracy on the same held-out (h, r) pairs.
#
# No pickles needed: `Predictions/{x}_intensities.csv` already tabulates the
# shipped models' predictions next to the LES reference, written by
# `fitModel.py <xlist> Write`.  Columns are `u LES, Iu LES, ...` and
# `u GPR, Iu GPR, ...` for all 28 pairs on a 150-point y grid, with `u`
# re-dimensionalised by 15 m/s and `y` in metres.
#
# Run from the repository root (`InflowGeneration/`).

# %%
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from hyperparametersGPR import trainPairs, devPairs, testPairs, xList  # noqa
import podlib as pl                                                    # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)


def show(df, name=None):
    if name:
        print(f'--- {name} ---')
    print(df.to_string())
    return df

QOIS = ['u', 'Iu', 'Iv', 'Iw']
Y_GRID = np.linspace(0.01, 1.0, 150)
REFVEL = 15.0

# Stations with a Predictions/ dump.
X_LIST = [x for x in xList
          if os.path.exists(f'Predictions/{str(x).replace(".", "p")}_intensities.csv')]
print(f'{len(X_LIST)} stations with shipped predictions: {X_LIST}')

# %% [markdown]
# ## 1. Read the shipped predictions

# %%
ship = pd.concat(
    [pd.read_csv(f'Predictions/{str(x).replace(".", "p")}_intensities.csv')
     for x in X_LIST],
    ignore_index=True)
ship['h'] = ship['h'].round(2)
ship['r'] = ship['r'].round().astype(int)
print(ship.shape, list(ship.columns))

test_key = {(round(h, 2), int(r)) for h, r in testPairs}
is_test = [(h, r) in test_key for h, r in zip(ship['h'], ship['r'])]
ship_te = ship[is_test].copy()
print(f'test rows: {len(ship_te)}  '
      f'(= {len(testPairs)} pairs x {len(X_LIST)} x x {len(Y_GRID)} y)')

# %% [markdown]
# ## 2. Fit POD-GPR on the same training pairs

# %%
meta_tr, X_tr = pl.build_snapshots(trainPairs, X_LIST, QOIS, Y_GRID)
meta_te, X_te = pl.build_snapshots(testPairs, X_LIST, QOIS, Y_GRID)

surr = {q: pl.PODSurrogate(n_modes=5).fit(meta_tr, X_tr[q], Y_GRID)
        for q in QOIS}
pod_te = {q: surr[q].predict(meta_te) for q in QOIS}

# %% [markdown]
# ## 3. Align the two prediction sets
#
# The snapshots are non-dimensional (`u` divided by 15 m/s in `loadData`);
# the CSV re-dimensionalised `u`.  Undo that so both live in the same units.

# %%
recs = []
for i, row in meta_te.reset_index(drop=True).iterrows():
    m = ((ship_te['h'] == round(row['h'], 2)) &
         (ship_te['r'] == int(row['r'])) &
         (np.abs(ship_te['x'] - row['x']) < 1e-9))
    sub = ship_te[m].sort_values('y')
    assert len(sub) == len(Y_GRID), (row.to_dict(), len(sub))
    for q in QOIS:
        scale = REFVEL if q == 'u' else 1.0
        recs.append(pd.DataFrame({
            'h': row['h'], 'r': row['r'], 'x': row['x'], 'y': Y_GRID, 'QoI': q,
            'LES': sub[f'{q} LES'].to_numpy() / scale,
            'point-wise GPR': sub[f'{q} GPR'].to_numpy() / scale,
            'POD-GPR': pod_te[q][i],
        }))
cmp = pd.concat(recs, ignore_index=True)

# Sanity: the CSV's LES column must match the snapshots we rebuilt.
for q in QOIS:
    a = cmp.loc[cmp['QoI'] == q, 'LES'].to_numpy()
    b = X_te[q].ravel()
    print(f'{q:>3}  max |CSV LES - rebuilt snapshot| = {np.abs(a - b).max():.2e}')

# %% [markdown]
# ## 4. Test-set RMSE, head to head

# %%
COLS = ['point-wise GPR', 'POD-GPR']


def rmse(g, col):
    return float(np.sqrt(np.mean((g[col].to_numpy() - g['LES'].to_numpy()) ** 2)))


tbl = pd.DataFrame(
    [{c: rmse(cmp[cmp['QoI'] == q], c) for c in COLS} for q in QOIS],
    index=QOIS)
tbl['ratio'] = tbl['POD-GPR'] / tbl['point-wise GPR']
show(tbl, 'test-set RMSE')

# %% [markdown]
# ## 5. RMSE against $x$ -- the paper's Figure 5

# %%
per_x = pd.DataFrame([
    {'QoI': q, 'x': x,
     **{c: rmse(cmp[(cmp['QoI'] == q) & (cmp['x'] == x)], c) for c in COLS}}
    for q in QOIS for x in X_LIST])
show(per_x.pivot(index='x', columns='QoI', values='POD-GPR'),
     'POD-GPR RMSE by station')

fig, axes = plt.subplots(1, len(QOIS), figsize=(4.2 * len(QOIS), 4))
for a, q in zip(axes, QOIS):
    s = per_x[per_x['QoI'] == q]
    a.plot(s['x'], s['point-wise GPR'], 'o-', label='point-wise GPR')
    a.plot(s['x'], s['POD-GPR'], 's--', label='POD-GPR')
    a.set_xlabel('$x$ [m]'); a.set_title(q); a.set_xscale('log')
    a.grid(alpha=.3)
axes[0].set_ylabel('test-set RMSE')
axes[0].legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/accuracy_vs_x.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Cost

# %%
n_ship = len(X_LIST) * len(QOIS)
mb = sum(os.path.getsize(f'../GPRModels/{str(x).replace(".", "p")}_intensities_{q}.pkl')
         for x in X_LIST for q in QOIS
         if os.path.exists(
             f'../GPRModels/{str(x).replace(".", "p")}_intensities_{q}.pkl')) / 1e6

n_pod = len(QOIS) * surr[QOIS[0]].Phi.shape[1]
n_train_ship = len(trainPairs) * 100          # X_train_ is (2000, 3)
n_train_pod = len(meta_tr)

show(pd.DataFrame({
    'models': [n_ship, n_pod],
    'training points each': [n_train_ship, n_train_pod],
    'Cholesky cost ~N^3': [n_train_ship ** 3, n_train_pod ** 3],
    'disk [MB]': [round(mb), 0],
}, index=['point-wise GPR', 'POD-GPR']), 'cost')
