# %% [markdown]
# # Experiment 3 (route 2) -- NSGA-II vs exhaustive grid
#
# The original plan was to compare a grid sweep against the *published* Pareto
# front.  That is not admissible: re-scoring the published decision variables
# does not reproduce `*_pareto_front.csv` (see `04_identify_config.py`).  The
# published run used xList A and an `intensities_high*` model set rather than
# `intensities`, and even then a ~1% residual remains, with published alpha
# straying outside the bounds the committed code declares.
#
# So compare the two optimisers directly instead, on a byte-identical
# surrogate.  Everything is held fixed -- same models, same objective, same
# bounds, same alpha range -- and only the search algorithm differs.  That is a
# cleaner test of "did 128 x 220 converge" than any comparison against a run
# whose configuration we cannot reconstruct.
#
# Needs `dev/cache_high_noA.npz` (built by `optcompare.build_cache`).

# %%
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import load_target, dominates                        # noqa
import gridsearch as gs                                          # noqa
import optcompare as oc                                          # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)

FNAME = 'MRB_Cat_B'
QOIS = ['u', 'Iu', 'Iv', 'Iw']
CACHE = 'dev/cache_high_noA.npz'

# alpha widened past the committed [0.30, 0.70]: the published solutions reach
# 0.250 and 0.727, so the bounds in optimizeParameters.py are not the ones that
# were actually searched.
ALPHA_LO, ALPHA_HI = 0.25, 0.73
# 0.005 leaves the grid ~0.05% behind NSGA-II on Iu/Iv/Iw, which is pure
# discretisation: those objectives do not involve k, so alpha is their only
# continuous variable and NSGA-II searches it freely.  0.001 closes the gap
# exactly and is converged (0.0005 changes nothing).
ALPHA_STEP = 0.001
ALPHA_GRID = np.round(np.arange(ALPHA_LO, ALPHA_HI + 1e-9, ALPHA_STEP), 5)

VARS = {r'$h$': [0.035, 0.1649], r'$r$': [51.5, 92.49],
        r'$\alpha$': [ALPHA_LO, ALPHA_HI], r'$k$': [0.8, 1.8],
        r'$x$': [-0.49, len(gs.X_LIST) - 0.51]}

# %%
cache = oc.load_cache(CACHE)
target, header, Uref, yref = load_target(FNAME)
obj = oc.CachedObjective(cache, target, QOIS)

print(f'{len(cache["xList"])} stations, {len(cache["HR"])} (h,r) pairs '
      f'-> {len(cache["xList"]) * len(cache["HR"]):,} configurations')
print(f'alpha grid: {ALPHA_LO} .. {ALPHA_HI} step {ALPHA_STEP} '
      f'({len(ALPHA_GRID)} values)')

# %% [markdown]
# ## 1. Cache fidelity
#
# The cache must return what the pickles return.  Check a few configurations
# against a live `predict_mean`.

# %%
probe = [(0.08, 72.0, 2.1), (0.04, 52.0, 0.6), (0.16, 92.0, 13.0)]
worst = 0.0
for h, r, x in probe:
    gp = gs.station(x)
    for q in QOIS:
        gs.load_qoi(gp, x, q, testID='intensities_high_noA')
        live = gs.predict_mean(gp, [(h, r)])[0]
        cached = cache['Q'][x][q][obj.rows(h, r)]
        worst = max(worst, float(np.abs(live - cached).max()))
    del gp
print(f'max |live - cached| = {worst:.3e}   '
      f'(float32 storage, so ~1e-7 is expected)')

# %% [markdown]
# ## 2. NSGA-II, exactly the committed settings

# %%
t0 = time.time()
nsga_F, nsga_X = oc.run_nsga(obj, cache['xList'], VARS,
                             pop_size=128, n_gen=220, seed=1)
t_nsga = time.time() - t0
print(f'NSGA-II: {128 * 220:,} evaluations in {t_nsga:.1f}s '
      f'-> {len(nsga_F)} non-dominated solutions')

# %% [markdown]
# ## 3. Exhaustive grid, k solved analytically

# %%
t0 = time.time()
grid = oc.run_grid(obj, cache['xList'], ALPHA_GRID)
t_grid = time.time() - t0
print(f'grid: {len(grid):,} evaluations in {t_grid:.1f}s')

# %% [markdown]
# ## 4. Does the grid dominate NSGA-II?

# %%
G = grid[QOIS].to_numpy()
N = nsga_F[QOIS].to_numpy()
n_dom = np.array([dominates(G, f).sum() for f in N])

print(f'{int((n_dom > 0).sum())} of {len(N)} NSGA-II solutions are strictly '
      f'dominated by a grid point  ({100*(n_dom > 0).mean():.0f}%)')

# The reverse direction only means something against the grid's own front:
# the grid enumerates the whole space, so of course most of its 4.6M points
# are bad.  The question is whether NSGA-II found anything the grid did not.
survivors = N[n_dom == 0]
beats = np.array([dominates(survivors, g).any() for g in G]) if len(survivors) \
    else np.zeros(len(G), bool)
print(f'{len(survivors)} NSGA-II solutions survive; they strictly dominate '
      f'{int(beats.sum()):,} of {len(G):,} grid points')
print('  (a surviving solution is one the grid ties rather than beats -- '
      'expected wherever the two coincide)')

# %% [markdown]
# ## 5. Per-objective best

# %%
comp = pd.DataFrame({'NSGA-II': nsga_F[QOIS].min(), 'grid': grid[QOIS].min()})
comp['improvement'] = 1 - comp['grid'] / comp['NSGA-II']
print(comp.to_string())

# %% [markdown]
# ## 6. Fronts

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for a, (qa, qb) in zip(axes, [('u', 'Iu'), ('u', 'Iw'), ('Iu', 'Iv')]):
    a.scatter(grid[qa], grid[qb], s=1, alpha=.04, color='tab:grey',
              rasterized=True, label='grid (all)')
    sub = grid[[qa, qb]].to_numpy()
    keep = []
    for i in np.argsort(sub[:, 0]):
        if not keep or sub[i, 1] < sub[keep[-1], 1]:
            keep.append(i)
    a.plot(sub[keep, 0], sub[keep, 1], '-', color='tab:blue', lw=2,
           label='grid front')
    a.scatter(nsga_F[qa], nsga_F[qb], s=26, color='tab:red', zorder=5,
              label='NSGA-II')
    a.set_xlabel(f'RMSE {qa}'); a.set_ylabel(f'RMSE {qb}')
axes[0].legend(frameon=False)
fig.suptitle(f'{FNAME} -- identical surrogate, identical objective')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/{FNAME}_optimiser_fronts.png', dpi=140,
            bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Design landscape
#
# What NSGA-II cannot show: whether the optimum sits in a broad valley or on a
# spike.  A spike is unusable, because the built case carries manufacturing and
# mesh-discretisation error in `h`.

# %%
w = grid['u'] / grid['u'].min() + grid['Iu'] / grid['Iu'].min()
best = grid.loc[w.idxmin()]
print(best.to_string())

sl = grid[(grid['$x$'] == best['$x$']) &
          (np.isclose(grid[r'$\alpha$'], best[r'$\alpha$']))]
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for a, q in zip(axes, ['u', 'Iu']):
    piv = sl.pivot_table(index='$h$', columns='$r$', values=q)
    cs = a.contourf(piv.columns, piv.index, piv.values, levels=30)
    a.plot(best['$r$'], best['$h$'], 'r*', ms=16)
    a.set_xlabel('$r$'); a.set_ylabel('$h$ [m]'); a.set_title(f'RMSE {q}')
    fig.colorbar(cs, ax=a)
fig.suptitle(rf'$x$={best["$x$"]} m,  $\alpha$={best[r"$\alpha$"]:.3f}')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/{FNAME}_landscape.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Cost

# %%
print(pd.DataFrame({
    'evaluations': [128 * 220, len(grid)],
    'GPR predictions': [128 * 220 * len(QOIS), len(cache['xList']) *
                        len(cache['HR']) * len(QOIS)],
    'wall time [s]': [round(t_nsga, 1), round(t_grid, 1)],
    'result': ['sampled front', 'complete enumeration'],
}, index=['NSGA-II', 'grid']).to_string())
