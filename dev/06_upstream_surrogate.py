# %% [markdown]
# # Upstream surrogate indexed by row number
#
# The database labels the two upstream planes with pseudo-coordinates that
# locate them in the *downstream* domain's frame: x = -4.95 is the downstream
# inlet, x = -2.85 the end of the ALF box.  They carry no meaning in the
# upstream canopy, where the two planes are the same quantity sampled 7 rows
# apart -- the label r naming the ALF station and the inlet station lying at
# r - 7.
#
# Fitting one model per pseudo-station therefore splits data that belongs
# together.  Pooling the planes and indexing by the true row number n doubles
# the training set at no simulation cost, halves the number of models, and
# turns the 7-row offset from a hardwired structure into a query-time choice.

# %%
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from hyperparametersGPR import trainPairs, devPairs, testPairs   # noqa
import podlib as pl                                              # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)

# Upstream QoIs are the mean velocity and the Reynolds stresses; yMax = 1.5 m
# is the fitted range used for the inflow boundary condition.
QOIS = ['u', 'uu', 'vv', 'ww', 'uv']
Y_GRID = np.linspace(0.01, 1.0, 150)
YMAX = 1.5


def show(df, name=None):
    if name:
        print(f'--- {name} ---')
    print(df.to_string())
    return df


# %% [markdown]
# ## 1. Station inventory
#
# Confirm that pooling really does double the stations, and that the row
# indices interleave as expected.

# %%
inv = []
for h, r in np.concatenate([trainPairs, devPairs, testPairs]):
    for x in (pl.X_INLET, pl.X_ALF):
        inv.append({'h': h, 'r': int(r), 'x': x, 'n': pl.row_index(r, x),
                    'plane': 'inlet' if x == pl.X_INLET else 'alf'})
inv = pd.DataFrame(inv)
show(inv.groupby('h')['n'].apply(lambda s: sorted(s.unique())),
     'distinct row indices per h (was: 5 labels, now 10 stations)')

# %% [markdown]
# ## 2. Build the pooled snapshots

# %%
meta_tr, X_tr = pl.build_upstream_snapshots(trainPairs, QOIS, Y_GRID, YMAX)
meta_te, X_te = pl.build_upstream_snapshots(testPairs, QOIS, Y_GRID, YMAX)
print(f'training profiles: {len(meta_tr)}   (per-plane fitting would give '
      f'{len(meta_tr)//2})')
print(f'test profiles    : {len(meta_te)}')
show(meta_tr.head(6), 'meta')

# %% [markdown]
# ## 3. POD spectrum
#
# The pooled basis has to represent both planes at once.  If that fails, the
# spectrum will say so.

# %%
tbl = {}
for q in QOIS:
    _, _, _, energy = pl.pod(X_tr[q])
    tbl[q] = [pl.modes_needed(energy, t) for t in (1e-2, 1e-3, 1e-4)]
show(pd.DataFrame(tbl, index=['99%', '99.9%', '99.99%']), 'modes needed')

# %% [markdown]
# ## 4. Fit and score
#
# Features are (h, n): two dimensions, no log transform, the row index being
# uniformly spaced.

# %%
surr = {}
for q in QOIS:
    surr[q] = pl.PODSurrogate(n_modes=5, features=('h', 'n'), log_cols=()
                              ).fit(meta_tr, X_tr[q], Y_GRID)
    print(f'{q:>3}  truncation RMSE {surr[q].truncation_error(X_te[q]):.6f}'
          f'   full surrogate RMSE {surr[q].total_error(meta_te, X_te[q]):.6f}')

# %%
for q in QOIS:
    print(f'--- {q} ---')
    print(surr[q].length_scales().round(3).to_string())

# %% [markdown]
# ## 5. The offset is now a query, not a structure
#
# Any pair of stations can be requested from one model.  The 7-row separation
# that the boundary conditions happen to use is one choice among many.

# %%
h0, n0 = 0.08, 65.0
fig, axes = plt.subplots(1, len(QOIS), figsize=(3.6 * len(QOIS), 4.5),
                         sharey=True)
for a, q in zip(axes, QOIS):
    for dn, style in [(0, '-'), (7, '--'), (14, ':')]:
        p = surr[q].predict(pd.DataFrame({'h': [h0], 'n': [n0 + dn]}))[0]
        a.plot(p, Y_GRID, style, label=f'n = {int(n0 + dn)}')
    a.set_xlabel(q)
axes[0].set_ylabel('$y/y_{max}$')
axes[0].legend(frameon=False, title=f'h = {h0} m')
fig.suptitle('One model, arbitrary station offsets')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/upstream_offsets.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Test-set predictions

# %%
fig, axes = plt.subplots(1, len(QOIS), figsize=(3.6 * len(QOIS), 4.5),
                         sharey=True)
for a, q in zip(axes, QOIS):
    P = surr[q].predict(meta_te)
    for i, (ref, prd) in enumerate(zip(X_te[q], P)):
        a.fill_betweenx(Y_GRID, ref * .9, ref * 1.1, color='tab:grey',
                        alpha=.2, label=r'LES $\pm$10%' if i == 0 else None)
        a.plot(ref, Y_GRID, color='tab:grey', lw=1.2)
        a.plot(prd, Y_GRID, '--', lw=1.3,
               label='POD-GPR' if i == 0 else None)
    a.set_xlabel(q)
axes[0].set_ylabel('$y/y_{max}$')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Upstream surrogate, test set (both planes pooled)')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/upstream_testset.png', dpi=140, bbox_inches='tight')
plt.show()
