"""Export everything the paper's data figures need into a single .mat.

MATLAB cannot read scikit-learn pickles, so the split is: Python evaluates the
surrogates and writes the numbers; MATLAB owns the drawing.  Run this whenever
the surrogates or the optimizer change, then re-run the MATLAB scripts.

    python dev/export_figdata.py

Writes ../Inflow_Paper/matlab/figdata.mat with one struct per figure.
"""

import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperparametersGPR import trainPairs, devPairs, testPairs, xList  # noqa
import podlib as pl                                                    # noqa
import optcompare as oc                                                # noqa
from common import load_target                                         # noqa

OUT = '../Inflow_Paper/matlab/figdata.mat'
Y = np.linspace(0.01, 1.0, 150)
QD = ['u', 'Iu', 'Iv', 'Iw']          # downstream QoIs
QU = ['u', 'uu', 'vv', 'ww', 'uv']    # upstream QoIs
XS = list(xList)
# P0-2: the POD+GPR cache built by 11_rebuild_cache_pod.py.  The former
# cache_high_noA.npz came from the superseded point-wise pickles, so
# figure 9 disagreed with the method the paper describes (HANDOVER 11.1).
CACHE = 'dev/cache_pod.npz'
CATS = ['B', 'C', 'D']
BLDG = ['LRB', 'MRB', 'HRB']
SELECT = ['u', 'Iu', 'Iw']            # I_vert is unreachable; see fig 9 notes
ALPHA_GRID = np.round(np.arange(0.25, 0.7301, 0.001), 5)

data = {'y': Y, 'xList': np.array(XS, dtype=float),
        'qoiD': np.array(QD, dtype=object),
        'qoiU': np.array(QU, dtype=object)}


# -- figure 3: overview of the downstream database --------------------------
print('fig3: database envelope and slices')
allPairs = np.concatenate([trainPairs, devPairs, testPairs])
meta_all, X_all = pl.build_snapshots(allPairs, XS, QD, Y)
h_all = meta_all['h'].to_numpy()
r_all = meta_all['r'].to_numpy()
x_all = meta_all['x'].to_numpy()

env_lo = np.stack([X_all[q].min(axis=0) for q in QD])
env_hi = np.stack([X_all[q].max(axis=0) for q in QD])

# (b) every (h, r) at one station
selb = np.isclose(x_all, 0.6)
# (c) two roughness heights swept in x at fixed r
selc = np.isin(h_all, [0.04, 0.12]) & np.isclose(r_all, 92)

data['fig3'] = {
    'env_lo': env_lo, 'env_hi': env_hi,
    'b_prof': np.stack([X_all[q][selb] for q in QD]),
    'b_h': h_all[selb], 'b_r': r_all[selb],
    'c_prof': np.stack([X_all[q][selc] for q in QD]),
    'c_h': h_all[selc], 'c_x': x_all[selc],
}

# -- figure 4: upstream surrogate on the test set ---------------------------
print('fig4: upstream test set')
mtr_u, Xtr_u = pl.build_upstream_snapshots(trainPairs, QU, Y, 1.5)
mte_u, Xte_u = pl.build_upstream_snapshots(testPairs, QU, Y, 1.5)
su = {q: pl.PODSurrogate(n_modes=5, features=('h', 'n'), log_cols=()
                         ).fit(mtr_u, Xtr_u[q], Y) for q in QU}
data['fig4'] = {
    'les': np.stack([Xte_u[q] for q in QU]),
    'sur': np.stack([su[q].predict(mte_u) for q in QU]),
    'h': mte_u['h'].to_numpy(), 'n': mte_u['n'].to_numpy(),
    'plane': np.array(mte_u['plane'].to_numpy(), dtype=object),
}

# -- figures 5 and 6: downstream surrogate ----------------------------------
print('fig5, fig6: downstream test set and RMSE')
mtr, Xtr = pl.build_snapshots(trainPairs, XS, QD, Y)
mte, Xte = pl.build_snapshots(testPairs, XS, QD, Y)
sd = {q: pl.PODSurrogate(n_modes=5).fit(mtr, Xtr[q], Y) for q in QD}
pred = {q: sd[q].predict(mte) for q in QD}

# rmse per (QoI, h, x) over the test pairs
h_te = np.unique(mte['h'].to_numpy())
rmse = np.full((len(QD), len(h_te), len(XS)), np.nan)
for iq, q in enumerate(QD):
    for ih, hv in enumerate(h_te):
        for ix, xv in enumerate(XS):
            m = np.isclose(mte['h'].to_numpy(), hv) & np.isclose(mte['x'].to_numpy(), xv)
            if m.any():
                rmse[iq, ih, ix] = np.sqrt(np.mean((pred[q][m] - Xte[q][m]) ** 2))
data['fig5'] = {'rmse': rmse, 'h': h_te, 'x': np.array(XS, dtype=float)}

show_x = [0.6, 2.1, 4.0]
sel6 = [np.flatnonzero(np.isclose(mte['x'].to_numpy(), xv)) for xv in show_x]
data['fig6'] = {
    'x': np.array(show_x, dtype=float),
    'les': np.stack([np.stack([Xte[q][s] for s in sel6]) for q in QD]),
    'sur': np.stack([np.stack([pred[q][s] for s in sel6]) for q in QD]),
    'h': np.stack([mte['h'].to_numpy()[s] for s in sel6]),
    'r': np.stack([mte['r'].to_numpy()[s] for s in sel6]),
}

# -- figure 9: ASCE targets and the optimised setups ------------------------
print('fig9: ASCE optima')
cache = oc.load_cache(CACHE)
f9 = {'bldg': np.array(BLDG, dtype=object), 'cat': np.array(CATS, dtype=object)}
eta_len = None
tg, pr, tab = [], [], []
for b in BLDG:
    for c in CATS:
        t, _, U_ref, y_T, y_ref = load_target(f'{b}_Cat_{c}')
        obj = oc.CachedObjective(cache, t, QD)
        g = oc.run_grid(obj, cache['xList'], ALPHA_GRID)
        # Selection rule, HANDOVER 11.5: min-normalised sum over SELECT.
        # Paper section 5.5 (TODO P0-6) has to state this as an equation.
        g['score'] = sum(g[q] / g[q].min() for q in SELECT)
        best = g.loc[g['score'].idxmin()]
        M = obj.M(best[r'$\alpha$'])
        i = obj.rows(best['$h$'], best['$r$'])
        p = []
        for q in QD:
            v = M @ cache['Q'][best['$x$']][q][i]
            p.append(v * best[r'$k$'] if q == 'u' else v)
        tg.append(np.stack([t[q].to_numpy() for q in QD]))
        pr.append(np.stack(p))
        eta_len = obj.eta
        nt = {q: np.linalg.norm(obj.T[q]) for q in QD}
        tab.append([best['$h$'], best['$r$'], best['$x$'], best[r'$\alpha$'],
                    best[r'$k$'], y_ref, U_ref, best[r'$k$'] * U_ref]
                   + [100 * best[q] / nt[q] for q in QD])
f9.update({'eta': eta_len, 'target': np.stack(tg), 'pred': np.stack(pr),
           'table': np.array(tab),
           'tablecols': np.array(['h', 'r', 'x_B', 'y_Max', 's_U', 'y_ref',
                                  'U_ref', 'V'] + [f'res_{q}' for q in QD],
                                dtype=object)})
data['fig9'] = f9

os.makedirs(os.path.dirname(OUT), exist_ok=True)
savemat(OUT, data, do_compression=True)
print(f'\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)')
for k in ['fig3', 'fig4', 'fig5', 'fig6', 'fig9']:
    print(f'  {k}: ' + ', '.join(
        f'{kk}{np.shape(vv)}' for kk, vv in data[k].items()))
