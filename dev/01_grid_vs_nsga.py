# %% [markdown]
# # Experiment 3 -- exhaustive grid vs NSGA-II
#
# Question: on the *same* surrogate and the *same* objective, does a complete
# sweep of the discrete design space find solutions that dominate the published
# Pareto front?
#
# If yes, the 128 x 220 NSGA-II run under-converged and the optimisation step
# can be replaced by something both cheaper and exhaustive.
#
# Run from the repository root (`InflowGeneration/`), with the `intensities`
# pkl models in `../GPRModels/`.
#
# In JupyterLab: right-click this file -> Open With -> Notebook (needs jupytext).

# %%
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import Y_GRID, load_target, check_alpha_operator, dominates  # noqa
import gridsearch as gs                                                   # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)


def show(df, name=None):
    """Print a frame -- readable both in a notebook and from the CLI."""
    if name:
        print(f'--- {name} ---')
    print(df.to_string())
    return df

FNAME = 'MRB_Cat_B'          # any of {LRB,MRB,HRB}_Cat_{B,C,D}
QOIS = ['u', 'Iu', 'Iv', 'Iw']
ALPHA_GRID = np.round(np.arange(0.30, 0.7001, 0.005), 4)

print(f'{len(gs.H_GRID)} h x {len(gs.R_GRID)} r x {len(gs.X_LIST)} x '
      f'= {len(gs.H_GRID)*len(gs.R_GRID)*len(gs.X_LIST)} configurations')
print(f'x {len(ALPHA_GRID)} alpha = '
      f'{len(gs.H_GRID)*len(gs.R_GRID)*len(gs.X_LIST)*len(ALPHA_GRID):,} '
      f'objective evaluations (k solved in closed form)')

# %% [markdown]
# ## 1. Target and the published NSGA-II result

# %%
target, header, U_ref, y_T, y_ref = load_target(FNAME)
print(f'target spans y/y_T in [{target["y"].min():.4f}, {target["y"].max():.4f}]'
      f'  ({len(target)} points),  y_T = {y_T:.3f} m,  '
      f'y_ref = {y_ref:.3f} m,  U(y_ref) = {U_ref:.3f} m/s')

nsga_F = pd.read_csv(f'TestCases/{FNAME}_pareto_front.csv')
nsga_X = pd.read_csv(f'TestCases/{FNAME}_decision_variables.csv')
print(f'\npublished front: {len(nsga_F)} solutions, objectives {list(nsga_F.columns)}')
show(nsga_F.describe().loc[['min', 'mean', 'max']], 'published front')

# %% [markdown]
# ## 2. Fidelity check
#
# The sweep replaces `scale_predictions` with a precomputed interpolation
# matrix. Confirm the two agree to machine precision before trusting anything
# downstream.

# %%
rng = np.random.default_rng(0)
fake = np.cumsum(rng.random(len(Y_GRID))) / len(Y_GRID)      # smooth, monotone
worst = check_alpha_operator(fake, target['y'].to_numpy(),
                             alphas=[0.30, 0.37, 0.5, 0.63, 0.70])
print(f'max |alpha_operator - scale_predictions| = {worst:.3e}')

# %% [markdown]
# ## 3. Reproduce the published objective values
#
# **This gate matters.** If re-scoring the published decision variables does not
# return the published Pareto front, the two runs are not using the same
# objective and no dominance claim is admissible.
#
# Each pickle is ~72 MB (a 3000x3000 Cholesky factor), so models are attached
# one at a time rather than preloaded the way MyProblem does it.

# %%
check = gs.reevaluate(target, QOIS, nsga_X.head(12))
side = pd.concat([nsga_F.head(12).add_suffix(' published'),
                  check.add_suffix(' re-scored')], axis=1)
resid = np.abs(check.to_numpy() - nsga_F.head(12).to_numpy()).max()
print(f'max |published - re-scored| = {resid:.3e}')
show(side, 're-scoring check')

# %% [markdown]
# ## 4. Sweep the full grid
#
# The first station also runs check_predict_mean(), asserting that the
# mean-only fast path reproduces gaussianProcess.predict's y_model exactly.

# %%
grid = gs.sweep(target, QOIS, ALPHA_GRID)
print(f'\n{len(grid):,} evaluated points')
show(grid.head(), 'grid sample')

# %% [markdown]
# ## 5. Does the grid dominate the published front?

# %%
G = grid[QOIS].to_numpy()
n_dom = np.array([dominates(G, f).sum() for f in nsga_F[QOIS].to_numpy()])

print(f'{np.sum(n_dom > 0)} of {len(nsga_F)} published solutions are dominated '
      f'by at least one grid point')
print(f'median number of dominating grid points: {np.median(n_dom):.0f}')

worst_i = int(np.argmax(n_dom))
if n_dom[worst_i] > 0:
    best = grid.iloc[np.argmin(np.linalg.norm(
        G - nsga_F[QOIS].to_numpy()[worst_i], axis=1)[dominates(
            G, nsga_F[QOIS].to_numpy()[worst_i])])]
    print('\nmost-dominated published solution:')
    print(nsga_F.iloc[worst_i].to_string())

# %% [markdown]
# ## 6. Per-objective best

# %%
comp = pd.DataFrame({
    'NSGA-II best': nsga_F[QOIS].min(),
    'grid best': grid[QOIS].min(),
})
comp['improvement'] = 1 - comp['grid best'] / comp['NSGA-II best']
show(comp, 'per-objective best')

# %% [markdown]
# ## 7. Two-objective view (u, Iu) -- the pair the paper actually optimises

# %%
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(grid['u'], grid['Iu'], s=1, alpha=0.05, color='tab:grey',
           label='grid (all points)', rasterized=True)

sub = grid[['u', 'Iu']].to_numpy()
keep = []
for i in np.argsort(sub[:, 0]):
    if not keep or sub[i, 1] < sub[keep[-1], 1]:
        keep.append(i)
ax.plot(sub[keep, 0], sub[keep, 1], '-', color='tab:blue', lw=2,
        label='grid Pareto front')

ax.scatter(nsga_F['u'], nsga_F['Iu'], s=28, color='tab:red', zorder=5,
           label='NSGA-II published front')

ax.set_xlabel('RMSE $u$')
ax.set_ylabel('RMSE $I_u$')
ax.set_title(FNAME)
ax.legend(frameon=False)
plt.savefig(f'{FIGDIR}/{FNAME}_pareto.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Design landscape -- is the optimum sharp or flat?
#
# NSGA-II cannot show this. A minimum sitting on a spike is not usable in
# practice, because the built case carries manufacturing and mesh discretisation
# error in `h`.

# %%
w = grid['u'] / grid['u'].min() + grid['Iu'] / grid['Iu'].min()
best_row = grid.loc[w.idxmin()]
print(best_row.to_string())

sl = grid[(grid['$x$'] == best_row['$x$']) &
          (np.isclose(grid[r'$\alpha$'], best_row[r'$\alpha$']))]
piv = sl.pivot_table(index='$h$', columns='$r$', values='u')

fig, ax = plt.subplots(figsize=(9, 4))
cs = ax.contourf(piv.columns, piv.index, piv.values, levels=30)
ax.plot(best_row['$r$'], best_row['$h$'], 'r*', ms=16)
ax.set_xlabel('$r$')
ax.set_ylabel('$h$ [m]')
ax.set_title(f'RMSE $u$ at $x$={best_row["$x$"]}, '
             rf'$\alpha$={best_row[r"$\alpha$"]:.3f}')
fig.colorbar(cs, ax=ax)
plt.savefig(f'{FIGDIR}/{FNAME}_landscape.png', dpi=140, bbox_inches='tight')
plt.show()
