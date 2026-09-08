# %% [markdown]
# # Old GPR vs new POD-GPR vs LES, on the held-out test set
#
# Two figures for circulation: one per surrogate family, one panel per QoI,
# with the LES reference, the shipped point-wise GPR and the POD-GPR drawn
# together.
#
# Colour encodes the method, not the case, so the reader sees grey = truth,
# one colour = old, one colour = new.  All four held-out (h, r) pairs are shown
# in each panel; the grey band is the LES +-10% the paper reports against.

# %%
import gc
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from hyperparametersGPR import trainPairs, testPairs, xList    # noqa
import podlib as pl                                            # noqa
import gridsearch as gs                                        # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)

Y_GRID = np.linspace(0.01, 1.0, 150)
REFVEL = 15.0

C_LES, C_OLD, C_NEW = '0.35', 'tab:red', 'tab:blue'
LBL = {'u': r'$U/U_\infty$', 'uu': r"$\overline{u'u'}/U_\infty^2$",
       'vv': r"$\overline{v'v'}/U_\infty^2$",
       'ww': r"$\overline{w'w'}/U_\infty^2$",
       'uv': r"$\overline{u'v'}/U_\infty^2$",
       'Iu': r'$I_{stream}$', 'Iv': r'$I_{vert}$', 'Iw': r'$I_{span}$'}


def rel_rmse(pred, ref):
    return 100 * np.sqrt(np.mean((pred - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))


def panel(ax, ref, old, new, qoi):
    """One QoI: LES band + every test case for each of the three sources."""
    for i in range(len(ref)):
        ax.fill_betweenx(Y_GRID, ref[i] * .9, ref[i] * 1.1, color=C_LES,
                         alpha=.13, lw=0,
                         label=r'LES $\pm$10\%' if i == 0 else None)
        ax.plot(ref[i], Y_GRID, '-', color=C_LES, lw=1.6,
                label='LES' if i == 0 else None)
        ax.plot(old[i], Y_GRID, ':', color=C_OLD, lw=1.5,
                label='GPR (point-wise)' if i == 0 else None)
        ax.plot(new[i], Y_GRID, '--', color=C_NEW, lw=1.5,
                label='POD-GPR' if i == 0 else None)
    ax.set_xlabel(LBL.get(qoi, qoi))
    ax.set_title(f'{LBL.get(qoi, qoi)}\n'
                 f'old {rel_rmse(old, ref):.1f}\\%   '
                 f'new {rel_rmse(new, ref):.1f}\\%', fontsize=10)


# %% [markdown]
# ## Downstream surrogate
#
# The shipped predictions are read from `Predictions/`, which `fitModel.py Write`
# tabulates next to the LES reference, so no pickles are needed here.

# %%
QOIS_D = ['u', 'Iu', 'Iv', 'Iw']
SHOW_X = [0.6, 2.1, 4.0]                     # the stations of the paper's Fig. 6

X_AVAIL = [x for x in xList if os.path.exists(
    f'Predictions/{str(x).replace(".", "p")}_intensities.csv')]
ship = pd.concat([pd.read_csv(f'Predictions/{str(x).replace(".", "p")}'
                              f'_intensities.csv') for x in X_AVAIL],
                 ignore_index=True)
ship['h'] = ship['h'].round(2)
ship['r'] = ship['r'].round().astype(int)

meta_tr, X_tr = pl.build_snapshots(trainPairs, X_AVAIL, QOIS_D, Y_GRID)
meta_te, X_te = pl.build_snapshots(testPairs, X_AVAIL, QOIS_D, Y_GRID)
surr_d = {q: pl.PODSurrogate(n_modes=5, features=('h', 'r', 'x'),
                             log_cols=('x',)).fit(meta_tr, X_tr[q], Y_GRID)
          for q in QOIS_D}
pod_d = {q: surr_d[q].predict(meta_te) for q in QOIS_D}

fig, axes = plt.subplots(len(SHOW_X), len(QOIS_D),
                         figsize=(3.5 * len(QOIS_D), 3.9 * len(SHOW_X)),
                         sharey=True)
for i, x in enumerate(SHOW_X):
    sel = np.flatnonzero(np.isclose(meta_te['x'].to_numpy(), x))
    for j, q in enumerate(QOIS_D):
        old = np.empty((len(sel), len(Y_GRID)))
        for k, idx in enumerate(sel):
            m = ((ship['h'] == round(meta_te['h'][idx], 2)) &
                 (ship['r'] == int(meta_te['r'][idx])) &
                 (np.abs(ship['x'] - x) < 1e-9))
            sub = ship[m].sort_values('y')
            old[k] = sub[f'{q} GPR'].to_numpy() / (REFVEL if q == 'u' else 1.0)
        panel(axes[i, j], X_te[q][sel], old, pod_d[q][sel], q)
        if j == 0:
            axes[i, j].set_ylabel(f'$x_B$ = {x} m\n$y/y_{{max}}$')
axes[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('Downstream surrogate $\\mathcal{S}^D$ -- held-out test set '
             '(4 unseen $(h,r)$ pairs per panel)', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/compare_downstream.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Upstream surrogate
#
# Old: one point-wise model per pseudo-station.  New: one pooled model indexed
# by the true row number.  Rows of the figure are the two planes.

# %%
QOIS_U = ['u', 'uu', 'vv', 'ww', 'uv']
YMAX_U = 1.5

meta_tr_u, X_tr_u = pl.build_upstream_snapshots(trainPairs, QOIS_U, Y_GRID, YMAX_U)
meta_te_u, X_te_u = pl.build_upstream_snapshots(testPairs, QOIS_U, Y_GRID, YMAX_U)
surr_u = {q: pl.PODSurrogate(n_modes=5, features=('h', 'n'), log_cols=()
                             ).fit(meta_tr_u, X_tr_u[q], Y_GRID)
          for q in QOIS_U}
pod_u = {q: surr_u[q].predict(meta_te_u) for q in QOIS_U}

plane_of = np.where(meta_te_u['plane'].to_numpy() == 'inlet',
                    pl.X_INLET, pl.X_ALF)
old_u = {q: np.empty_like(X_te_u[q]) for q in QOIS_U}
for x in (pl.X_INLET, pl.X_ALF):
    gp = gs.station(x, YMAX_U, './GPRDatabase')
    for q in QOIS_U:
        gs.load_qoi(gp, x, q, testID='inflow_stresses')
        for i in np.flatnonzero(np.isclose(plane_of, x)):
            old_u[q][i] = gs.predict_mean(
                gp, [(meta_te_u['h'][i], meta_te_u['r'][i])], yGrid=Y_GRID)[0]
    gp.predictive_model = None
    del gp
    gc.collect()

PLANES = [('inlet', 'inlet plane ($n=r-7$)'), ('alf', 'ALF plane ($n=r$)')]
fig, axes = plt.subplots(len(PLANES), len(QOIS_U),
                         figsize=(3.5 * len(QOIS_U), 3.9 * len(PLANES)),
                         sharey=True)
for i, (tag, title) in enumerate(PLANES):
    sel = np.flatnonzero(meta_te_u['plane'].to_numpy() == tag)
    for j, q in enumerate(QOIS_U):
        panel(axes[i, j], X_te_u[q][sel], old_u[q][sel], pod_u[q][sel], q)
        if j == 0:
            axes[i, j].set_ylabel(f'{title}\n$y/y_{{max}}$')
axes[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('Upstream surrogate $\\mathcal{S}^U$ -- held-out test set '
             '(4 unseen $(h,r)$ pairs per panel)', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/compare_upstream.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Summary table

# %%
rows = []
for q in QOIS_D:
    old = np.empty_like(X_te[q])
    for i in range(len(meta_te)):
        m = ((ship['h'] == round(meta_te['h'][i], 2)) &
             (ship['r'] == int(meta_te['r'][i])) &
             (np.abs(ship['x'] - meta_te['x'][i]) < 1e-9))
        old[i] = ship[m].sort_values('y')[f'{q} GPR'].to_numpy() / \
            (REFVEL if q == 'u' else 1.0)
    rows.append({'surrogate': 'downstream', 'QoI': q,
                 'old [%]': rel_rmse(old, X_te[q]),
                 'new [%]': rel_rmse(pod_d[q], X_te[q])})
for q in QOIS_U:
    rows.append({'surrogate': 'upstream', 'QoI': q,
                 'old [%]': rel_rmse(old_u[q], X_te_u[q]),
                 'new [%]': rel_rmse(pod_u[q], X_te_u[q])})
tbl = pd.DataFrame(rows)
tbl['old/new'] = tbl['old [%]'] / tbl['new [%]']
print(tbl.round(2).to_string(index=False))
