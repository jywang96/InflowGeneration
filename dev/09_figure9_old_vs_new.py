# %% [markdown]
# # Figure 9 redrawn: published setups vs enumerated optima
#
# The paper's Figure 9 compares the optimizer's chosen setup against the nine
# ASCE target ABLs.  This version overlays both: the setups tabulated in the
# paper, and the optima found by enumerating the design grid with $s_U$ solved
# in closed form.  Both are evaluated with the *same* surrogate, so the
# comparison isolates the choice of setup.
#
# Two caveats that the figure marks explicitly:
#
#   * Four of the nine published setups use $r < 52$, below the training range
#     $r\in[52,92]$ of the database, and several use $y^{Max}<0.30$, below the
#     bound declared in the committed optimizer.  Evaluating them requires the
#     surrogate to extrapolate; those panels are flagged.
#   * The comparison is at surrogate level.  Neither setup's ABL has been
#     verified by LES in this script.

# %%
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import load_target, alpha_operator, optimal_k, dominates   # noqa
import gridsearch as gs                                                # noqa
import optcompare as oc                                                # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)

QOIS = ['u', 'Iu', 'Iv', 'Iw']
LBL = {'u': r'$U/U(y^T)$', 'Iu': r'$I_{stream}$',
       'Iv': r'$I_{vert}$', 'Iw': r'$I_{span}$'}
CACHE = 'dev/cache_high_noA.npz'
R_MIN, R_MAX = 52.0, 92.0          # training envelope of the database

# Table "Optimal simulation setup", as published.
PUBLISHED = pd.DataFrame([
    ('LRB', 'B', 0.07, 74, 0.6, 1.62, 0.34),
    ('LRB', 'C', 0.04, 62, 1.5, 1.53, 0.26),
    ('LRB', 'D', 0.04, 79, 3.6, 1.54, 0.28),
    ('MRB', 'B', 0.04, 67, 0.6, 1.60, 0.25),
    ('MRB', 'C', 0.04, 45, 3.6, 1.32, 0.31),
    ('MRB', 'D', 0.04, 53, 7.0, 1.36, 0.30),
    ('HRB', 'B', 0.04, 50, 1.8, 1.30, 0.38),
    ('HRB', 'C', 0.04, 48, 9.0, 1.33, 0.25),
    ('HRB', 'D', 0.04, 51, 13.0, 1.30, 0.35),
], columns=['bldg', 'terrain', 'h', 'r', 'x_B', 's_U', 'y_Max'])
# 'cat' would shadow the pandas .cat accessor on itertuples/rows
PUBLISHED['extrapolates'] = PUBLISHED['r'] < R_MIN

ALPHA_GRID = np.round(np.arange(0.25, 0.7301, 0.001), 5)

# %% [markdown]
# ## 1. Published setups: how many need extrapolation?

# %%
print(PUBLISHED.to_string(index=False))
print(f'\n{PUBLISHED["extrapolates"].sum()} of {len(PUBLISHED)} setups have '
      f'r < {R_MIN:.0f}, i.e. below the database training range')

# %% [markdown]
# ## 2. Enumerate each target
#
# One cache serves all nine targets: it holds surrogate profiles on the
# $(h,r,x_B)$ grid and is independent of the target, which enters only through
# the objective.

# %%
cache = oc.load_cache(CACHE)
Y_INT = np.array(oc.load_cache(CACHE)['HR'])          # touch, keeps linters quiet

results = {}
for _, row in PUBLISHED.iterrows():
    fname = f'{row.bldg}_Cat_{row.terrain}'
    target, _, U_ref, y_T, y_ref = load_target(fname)
    obj = oc.CachedObjective(cache, target, QOIS)

    grid = oc.run_grid(obj, cache['xList'], ALPHA_GRID)

    # Selection rule, applied identically for every case: among the Pareto set
    # of the two QoIs the paper optimises, take the one minimising the sum of
    # residuals normalised by the best achievable value of each.
    w = grid['u'] / grid['u'].min() + grid['Iu'] / grid['Iu'].min()
    best = grid.loc[w.idxmin()]

    # Score the published setup with the same surrogate.  h and r must be on
    # the cache grid; r < 52 is outside it, so fall back to a live prediction.
    eta = target['y'].to_numpy()
    M = alpha_operator(row.y_Max, eta)
    pub_pred, pub_res = {}, {}
    for q in QOIS:
        try:
            i = obj.rows(row.h, row.r)
            prof = cache['Q'][row.x_B][q][i]
        except KeyError:
            gp = gs.station(row.x_B)
            gs.load_qoi(gp, row.x_B, q, testID='intensities_high_noA')
            prof = gs.predict_mean(gp, [(row.h, row.r)])[0]
            gp.predictive_model = None
            del gp
        p = M @ prof
        if q == 'u':
            p = p * row.s_U
        pub_pred[q] = p
        pub_res[q] = float(np.linalg.norm(p - target[q].to_numpy()))

    new_pred = {}
    Mn = alpha_operator(best[r'$\alpha$'], eta)
    for q in QOIS:
        p = Mn @ cache['Q'][best['$x$']][q][obj.rows(best['$h$'], best['$r$'])]
        if q == 'u':
            p = p * best[r'$k$']
        new_pred[q] = p

    results[fname] = dict(target=target, pub_pred=pub_pred, pub_res=pub_res,
                          new_pred=new_pred, best=best, row=row)
    print(f'{fname:12s} published r={row.r:.0f}'
          f'{" (extrapolated)" if row.extrapolates else "":16s}'
          f'  ->  enumerated h={best["$h$"]:.2f} r={best["$r$"]:.0f} '
          f'x_B={best["$x$"]:.1f} y_Max={best[r"$\alpha$"]:.3f} '
          f's_U={best[r"$k$"]:.2f}')

# %% [markdown]
# ## 3. Residual comparison

# %%
rows = []
for fname, d in results.items():
    for q in QOIS:
        t = d['target'][q].to_numpy()
        rows.append({'case': fname, 'QoI': q,
                     'published [%]': 100 * d['pub_res'][q] / np.linalg.norm(t),
                     'enumerated [%]': 100 * np.linalg.norm(
                         d['new_pred'][q] - t) / np.linalg.norm(t),
                     'extrap': d['row'].extrapolates})
res = pd.DataFrame(rows)
print(res.pivot(index='case', columns='QoI',
                values=['published [%]', 'enumerated [%]']).round(2).to_string())

# %% [markdown]
# ## 4. The figure
#
# Rows are QoIs, columns are terrain categories, colour is building class --
# the layout of the paper's Figure 9.  Solid + band is the ASCE target and its
# $\pm10\%$; dotted is the published setup, dashed the enumerated one.

# %%
CATS = ['B', 'C', 'D']
BCOL = {'LRB': '#1f77b4', 'MRB': '#2ca02c', 'HRB': '#ff7f0e'}

fig, axes = plt.subplots(len(QOIS), len(CATS),
                         figsize=(4.6 * len(CATS), 3.7 * len(QOIS)),
                         sharey=True)
for i, q in enumerate(QOIS):
    for j, cat in enumerate(CATS):
        ax = axes[i, j]
        for bldg, col in BCOL.items():
            d = results[f'{bldg}_Cat_{cat}']
            eta = d['target']['y'].to_numpy()
            t = d['target'][q].to_numpy()
            ax.fill_betweenx(eta, t * .9, t * 1.1, color=col, alpha=.13, lw=0)
            ax.plot(t, eta, '-', color=col, lw=1.9,
                    label=f'{bldg} target' if (i == 0 and j == 0) else None)
            ax.plot(d['pub_pred'][q], eta, ':', color=col, lw=1.5,
                    label='published setup' if (i == 0 and j == 0 and
                                                bldg == 'LRB') else None)
            ax.plot(d['new_pred'][q], eta, '--', color=col, lw=1.5,
                    label='enumerated optimum' if (i == 0 and j == 0 and
                                                   bldg == 'LRB') else None)
        if i == 0:
            ax.set_title(f'Category {cat}', fontsize=11)
            flagged = [b for b in BCOL
                       if results[f'{b}_Cat_{cat}']['row'].extrapolates]
            if flagged:
                ax.text(.98, .02, 'published setup extrapolates in $r$:\n'
                        + ', '.join(flagged), transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=7.5, color='0.3',
                        bbox=dict(boxstyle='round,pad=.3', fc='#fff4e5',
                                  ec='0.6', lw=.7))
        if i == len(QOIS) - 1:
            ax.set_xlabel(LBL[q])
        else:
            ax.set_xlabel(LBL[q], fontsize=9)
        if j == 0:
            ax.set_ylabel(r'$y/y^{T}$')
        ax.set_ylim(0, 1)
axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle('ASCE 49-21 targets: published setup vs enumerated optimum, '
             'both evaluated with the same surrogate', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig9_old_vs_new.png', dpi=150, bbox_inches='tight')
plt.show()
