# %% [markdown]
# # Experiment 1 -- how compressible are the database profiles?
#
# Before replacing the point-wise GPR with a POD surrogate, establish the
# accuracy ceiling: how many POD modes are needed, and what is the
# reconstruction error with no regression involved at all?  If truncation alone
# already exceeds the +-10% the paper reports, the idea is dead and nothing
# downstream matters.
#
# Run from the repository root (`InflowGeneration/`).
# In JupyterLab: right-click -> Open With -> Notebook (needs jupytext).

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
Y_GRID = np.linspace(0.01, 1.0, 150)     # matches Predictions/*.csv
X_LIST = list(xList)                     # all 19 stations, 0.3 .. 13.0

print(f'{len(trainPairs)} train / {len(devPairs)} dev / {len(testPairs)} test '
      f'(h, r) pairs  x  {len(X_LIST)} stations')
print(f'-> {len(trainPairs)*len(X_LIST)} training profiles of {len(Y_GRID)} points')
print(f'   (the shipped surrogate instead sees '
      f'{len(trainPairs)}x100 = {len(trainPairs)*100} scalar points per (x, QoI))')

# %% [markdown]
# ## 1. Build the snapshot matrices
#
# ~1 min: one file read per (h, r), all stations extracted from it.

# %%
meta_tr, X_tr = pl.build_snapshots(trainPairs, X_LIST, QOIS, Y_GRID)
meta_dv, X_dv = pl.build_snapshots(devPairs,   X_LIST, QOIS, Y_GRID)
meta_te, X_te = pl.build_snapshots(testPairs,  X_LIST, QOIS, Y_GRID)

print({q: X_tr[q].shape for q in QOIS})

# %% [markdown]
# ## 2. POD spectrum
#
# The claim to test: ABL profiles are smooth and strongly family-like, so a
# handful of modes should carry essentially all the variance.

# %%
spectra = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for q in QOIS:
    _, _, _, energy = pl.pod(X_tr[q])
    spectra[q] = energy
    ax[0].semilogy(np.arange(1, 16), 1 - energy[:15], 'o-', label=q)
    ax[1].plot(np.arange(1, 16), energy[:15] * 100, 'o-', label=q)

ax[0].set_xlabel('modes retained'); ax[0].set_ylabel('unexplained variance')
ax[0].axhline(1e-3, ls='--', c='k', lw=1)
ax[0].legend(frameon=False)
ax[1].set_xlabel('modes retained'); ax[1].set_ylabel('cumulative energy [%]')
ax[1].set_ylim(95, 100.05); ax[1].grid(alpha=.3)
plt.savefig(f'{FIGDIR}/pod_spectrum.png', dpi=140, bbox_inches='tight')
plt.show()

show(pd.DataFrame({q: [pl.modes_needed(spectra[q], t) for t in (1e-2, 1e-3, 1e-4)]
                   for q in QOIS},
                  index=['99%', '99.9%', '99.99%']), 'modes needed')

# %% [markdown]
# ## 3. What do the modes look like?
#
# Mode 1 is usually overall amplitude; mode 2 a shift in boundary-layer
# thickness; mode 3 the near-ground canopy-wake bulge trading against the
# plateau aloft.  Section 4.1 of the paper describes these trends in words --
# POD quantifies them.

# %%
fig, axes = plt.subplots(1, len(QOIS), figsize=(4 * len(QOIS), 5), sharey=True)
for a, q in zip(axes, QOIS):
    qbar, Phi, _, _ = pl.pod(X_tr[q], 4)
    a.plot(qbar, Y_GRID, 'k', lw=2.5, label='mean')
    for k in range(4):
        a.plot(Phi[:, k], Y_GRID, lw=1.4, label=f'mode {k+1}')
    a.set_xlabel(q); a.axvline(0, c='k', lw=.5)
axes[0].set_ylabel('$y/y_{max}$'); axes[0].legend(frameon=False, fontsize=8)
plt.savefig(f'{FIGDIR}/pod_modes.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Truncation error -- the accuracy ceiling
#
# Project the held-out profiles onto the training basis and reconstruct.  No
# regression: this is purely "can the basis represent these shapes".

# %%
rows = []
for q in QOIS:
    qbar, Phi, _, _ = pl.pod(X_tr[q], 5)
    for name, X in [('train', X_tr[q]), ('dev', X_dv[q]), ('test', X_te[q])]:
        Xh = qbar + (X - qbar) @ Phi @ Phi.T
        rows.append({'QoI': q, 'set': name,
                     'RMSE': np.sqrt(np.mean((X - Xh) ** 2)),
                     'rel. RMSE [%]': 100 * np.sqrt(np.mean(((X - Xh) /
                                                             np.abs(X)) ** 2))})
show(pd.DataFrame(rows).pivot(index='QoI', columns='set',
                              values=['RMSE', 'rel. RMSE [%]']),
     'truncation error (no regression)')

# %% [markdown]
# ## 5. Fit the surrogate
#
# One GPR per mode, ARD Matern over (h, r, log x).  380 samples, 3-D input --
# seconds, versus the 128-core 256-seed search the shipped models needed.

# %%
surr = {}
for q in QOIS:
    surr[q] = pl.PODSurrogate(n_modes=5).fit(meta_tr, X_tr[q], Y_GRID)
    print(f'{q:>3}  truncation RMSE {surr[q].truncation_error(X_te[q]):.5f}'
          f'   full surrogate RMSE {surr[q].total_error(meta_te, X_te[q]):.5f}')

# %% [markdown]
# ## 6. Fitted ARD length scales
#
# The point the shipped kernels cannot express: the correlation length in `h`,
# in `r` and in `x` are unrelated quantities.  `RBF()` with a scalar
# length_scale forces them equal (after z-scoring).  Here they are free.

# %%
for q in QOIS:
    print(f'--- {q} ---')
    print(surr[q].length_scales().round(3).to_string())

# %% [markdown]
# ## 7. Test-set predictions

# %%
show_x = [0.6, 2.1, 4.0]                          # the stations in paper Fig. 6
fig, axes = plt.subplots(len(show_x), len(QOIS),
                         figsize=(4 * len(QOIS), 4 * len(show_x)), sharey=True)
for i, x in enumerate(show_x):
    sel = np.isclose(meta_te['x'].to_numpy(), x)
    for j, q in enumerate(QOIS):
        a = axes[i, j]
        P = surr[q].predict(meta_te[sel])
        for n, (ref, prd) in enumerate(zip(X_te[q][sel], P)):
            a.fill_betweenx(Y_GRID, ref * .9, ref * 1.1, color='tab:grey',
                            alpha=.25, label=r'LES $\pm$10%' if n == 0 else None)
            a.plot(ref, Y_GRID, color='tab:grey', lw=1.5)
            a.plot(prd, Y_GRID, '--', lw=1.6,
                   label='POD-GPR' if n == 0 else None)
        a.set_xlabel(f'{q}   ($x$={x} m)')
    axes[i, 0].set_ylabel('$y/y_{max}$')
axes[0, 0].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/pod_testset.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. `x` is now continuous
#
# The shipped surrogate can only answer at the 19 trained stations, which is
# why the optimiser carries `x` as an integer index.  Sample between them.

# %%
q = 'Iu'
xs = np.linspace(0.6, 4.0, 60)
grid = pd.DataFrame({'h': 0.10, 'r': 74.0, 'x': xs})
P = surr[q].predict(grid)

fig, ax = plt.subplots(figsize=(7, 5))
m = ax.pcolormesh(xs, Y_GRID, P.T, shading='auto')
for xt in [x for x in X_LIST if 0.6 <= x <= 4.0]:
    ax.axvline(xt, c='w', lw=.6, alpha=.6)
ax.set_xlabel('$x$ [m]   (white lines = trained stations)')
ax.set_ylabel('$y/y_{max}$')
ax.set_title(f'{q} at $h$=0.10 m, $r$=74')
fig.colorbar(m, ax=ax)
plt.savefig(f'{FIGDIR}/pod_x_continuous.png', dpi=140, bbox_inches='tight')
plt.show()
