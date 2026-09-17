# %% [markdown]
# # Figure 9: ASCE targets against the enumerated optima
#
# One curve per case, the setup chosen by enumerating the design grid with the
# velocity scale anchored at the building height.
#
# The selection rule optimises the three *reachable* quantities and reports the
# fourth as a diagnostic.  ASCE 49-21 prescribes
# $I_{vert}/I_{stream}=0.5$, whereas the downstream database spans $0.576$ to
# $1.116$ over its 380 profiles, with a 5th percentile of $0.618$: the required
# vertical anisotropy lies outside the reachable set entirely, so no setup, no
# surrogate and no optimizer can attain it.  Including it among the objectives
# only trades away accuracy on the three that are attainable -- doing so raises
# the worst reachable residual from 6.9% to 7.5%.

# %%
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import load_target                                   # noqa
import optcompare as oc                                          # noqa

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)

QOIS = ['u', 'Iu', 'Iv', 'Iw']
SELECT = ['u', 'Iu', 'Iw']                 # the reachable subset
LBL = {'u': r'$U/U^{T}(y_{ref})$', 'Iu': r'$I_{stream}$',
       'Iv': r'$I_{vert}$', 'Iw': r'$I_{span}$'}
CATS = ['B', 'C', 'D']
BCOL = {'LRB': '#1f77b4', 'MRB': '#2ca02c', 'HRB': '#ff7f0e'}
ALPHA_GRID = np.round(np.arange(0.25, 0.7301, 0.001), 5)

cache = oc.load_cache('dev/cache_high_noA.npz')

# %%
res, setups = {}, []
for b, col in BCOL.items():
    for c in CATS:
        f = f'{b}_Cat_{c}'
        t, _, U_ref, y_T, y_ref = load_target(f)
        obj = oc.CachedObjective(cache, t, QOIS)
        g = oc.run_grid(obj, cache['xList'], ALPHA_GRID)
        best = g.loc[sum(g[q] / g[q].min() for q in SELECT).idxmin()]

        eta = obj.eta
        M = obj.M(best[r'$\alpha$'])
        i = obj.rows(best['$h$'], best['$r$'])
        pred = {}
        for q in QOIS:
            p = M @ cache['Q'][best['$x$']][q][i]
            pred[q] = p * best[r'$k$'] if q == 'u' else p

        nt = {q: np.linalg.norm(obj.T[q]) for q in QOIS}
        res[f] = dict(target=t, pred=pred, eta=eta, best=best)
        setups.append({
            'case': f, 'y_ref [m]': y_ref, r'U(y_ref) [m/s]': U_ref,
            'h [m]': best['$h$'], 'r': int(best['$r$']),
            'x_B [m]': best['$x$'], 'y_Max [m]': best[r'$\alpha$'],
            's_U': best[r'$k$'], 'V [m/s]': best[r'$k$'] * U_ref,
            **{f'{q} [%]': 100 * best[q] / nt[q] for q in QOIS}})

setups = pd.DataFrame(setups).set_index('case')
print(setups.round(3).to_string())

# %%
fig, axes = plt.subplots(len(QOIS), len(CATS),
                         figsize=(4.6 * len(CATS), 3.6 * len(QOIS)),
                         sharey=True)
for i, q in enumerate(QOIS):
    for j, c in enumerate(CATS):
        ax = axes[i, j]
        for b, col in BCOL.items():
            d = res[f'{b}_Cat_{c}']
            t = d['target'][q].to_numpy()
            ax.fill_betweenx(d['eta'], t * .9, t * 1.1, color=col, alpha=.15, lw=0)
            ax.plot(t, d['eta'], '-', color=col, lw=1.9,
                    label=f'{b} target' if (i == 0 and j == 0) else None)
            ax.plot(d['pred'][q], d['eta'], '--', color=col, lw=1.6,
                    label='optimised setup' if (i == 0 and j == 0 and b == 'LRB')
                    else None)
        ax.axhline(2 / 3, color='0.5', lw=.8, ls=(0, (4, 3)))
        if i == 0:
            ax.set_title(f'Category {c}', fontsize=11)
        if q == 'Iv':
            ax.text(.97, .05, 'outside the reachable set:\n'
                    r'ASCE $I_{vert}/I_{stream}=0.50$,' '\n'
                    r'database $\geq 0.58$',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7.5, color='0.25',
                    bbox=dict(boxstyle='round,pad=.35', fc='#fff4e5',
                              ec='0.6', lw=.7))
        ax.set_xlabel(LBL[q])
        if j == 0:
            ax.set_ylabel(r'$y/y^{T}$')
        ax.set_ylim(0, 1)
axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
axes[0, 0].text(.03, .60, r'$\;\cdots\;y_{ref}$', transform=axes[0, 0].transAxes,
                fontsize=8, color='0.4')
fig.suptitle('ASCE 49-21 targets and the optimised downstream setups.  '
             r'Velocity anchored at $y_{ref}=H$; '
             r'$I_{vert}$ shown but not optimised.', fontsize=12, y=0.997)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig9_new.png', dpi=150, bbox_inches='tight')
plt.show()
