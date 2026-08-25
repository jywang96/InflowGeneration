"""NSGA-II vs exhaustive grid, on a byte-identical surrogate.

`MyProblem.eval_model_delayed` discretises its own decision variables on entry:

    h -> np.round(h*100)/100      over [0.035, 0.1649]  ->  13 values
    r -> np.round(r)              over [51.5,  92.49 ]  ->  41 values
    x -> xList[int(np.round(.))]                        ->  18 values

So every candidate NSGA-II can ever propose lands on the same 13*41*18 = 9594
point grid.  Evaluating that grid once and caching the profiles means both
optimisers read exactly the same numbers -- the comparison then isolates the
search algorithm and nothing else -- and turns each NSGA-II evaluation from
four GPR predictions into an array lookup.

Cache size: 18 * 533 * 399 * 4 QoI * 8 B ~ 122 MB, written to .npz so the
~20 min of GPR predictions is paid once.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymoo.algorithms.moo.nsga2 import NSGA2                     # noqa: E402
from pymoo.core.problem import Problem                           # noqa: E402
from pymoo.optimize import minimize                              # noqa: E402

from common import Y_GRID, alpha_operator, optimal_k             # noqa: E402
import gridsearch as gs                                          # noqa: E402


def build_cache(path, xList=gs.X_LIST, QoIs=gs.QOIS, hGrid=gs.H_GRID,
                rGrid=gs.R_GRID, testID='intensities_high_noA', verbose=True,
                **kw):
    """Predict every (h, r, x) profile once and store it in an .npz."""
    if os.path.exists(path):
        if verbose:
            print(f'reusing {path}')
        return load_cache(path)

    HR = np.array([(h, r) for h in hGrid for r in rGrid], dtype=float)
    store = {}
    for x in xList:
        t0 = time.time()
        _, Q = gs._profiles_for_x(x, QoIs, hGrid, rGrid, testID=testID, **kw)
        for q in QoIs:
            store[f'{x}|{q}'] = Q[q].astype(np.float32)
        if verbose:
            print(f'  x={x:<5} ({time.time() - t0:.1f}s)')

    np.savez_compressed(path, HR=HR, xList=np.array(xList),
                        QoIs=np.array(QoIs, dtype=object), **store)
    return load_cache(path)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    xList = [float(v) for v in z['xList']]
    QoIs = [str(v) for v in z['QoIs']]
    Q = {x: {q: z[f'{x}|{q}'].astype(float) for q in QoIs} for x in xList}
    return {'HR': z['HR'], 'xList': xList, 'QoIs': QoIs, 'Q': Q}


class CachedObjective:
    """Score any (h, r, x, alpha, k) against a target, from cached profiles."""

    def __init__(self, cache, target, QoIs, kBounds=(0.8, 1.8)):
        self.c = cache
        self.QoIs = list(QoIs)
        self.kBounds = kBounds
        self.eta = target['y'].to_numpy()
        self.T = {q: target[q].to_numpy() for q in self.QoIs}
        self._M = {}

        HR = cache['HR']
        self.hVals = np.unique(HR[:, 0])
        self.rVals = np.unique(HR[:, 1])
        # HR was built h-major, so this reproduces its row order
        self._row = {(round(h, 2), int(r)): i
                     for i, (h, r) in enumerate(HR)}

    def M(self, alpha):
        key = round(float(alpha), 6)
        if key not in self._M:
            self._M[key] = alpha_operator(key, self.eta)
        return self._M[key]

    def rows(self, h, r):
        return self._row[(round(float(h), 2), int(r))]

    def score(self, h, r, x, alpha, k=None):
        """Returns (objectives dict, k used).  k=None solves it in closed form."""
        i = self.rows(h, r)
        M = self.M(alpha)
        P = {q: M @ self.c['Q'][x][q][i] for q in self.QoIs}

        if k is None and 'u' in self.QoIs:
            kk, _ = optimal_k(P['u'][None, :], self.T['u'], *self.kBounds)
            k = float(kk[0])
        elif k is None:
            k = 1.0

        out = {}
        for q in self.QoIs:
            pred = k * P[q] if q == 'u' else P[q]
            out[q] = float(np.linalg.norm(pred - self.T[q]))
        return out, k


class CachedProblem(Problem):
    """MyProblem's objective, reading the cache instead of the pickles."""

    def __init__(self, obj, xList, varDict):
        self.obj = obj
        self.xList = list(xList)
        self.varNames = [r'$h$', r'$r$', r'$\alpha$', r'$k$', r'$x$']
        super().__init__(
            n_var=5, n_obj=len(obj.QoIs),
            xl=[varDict[n][0] for n in self.varNames],
            xu=[varDict[n][1] for n in self.varNames])

    def _evaluate(self, params, out, *args, **kwargs):
        F = np.empty((params.shape[0], len(self.obj.QoIs)))
        for i, p in enumerate(params):
            h = np.round(p[0] * 100) / 100
            r = np.round(p[1])
            x = self.xList[int(np.round(p[4]))]
            s, _ = self.obj.score(h, r, x, p[2], k=p[3])
            F[i] = [s[q] for q in self.obj.QoIs]
        out['F'] = F


def run_nsga(obj, xList, varDict, pop_size=128, n_gen=220, seed=1,
             verbose=False):
    prob = CachedProblem(obj, xList, varDict)
    res = minimize(prob, NSGA2(pop_size=pop_size), ('n_gen', n_gen),
                   seed=seed, verbose=verbose)
    F = pd.DataFrame(res.F, columns=obj.QoIs)
    X = pd.DataFrame(res.X, columns=prob.varNames)
    return F, X


def run_grid(obj, xList, alphaGrid, hGrid=gs.H_GRID, rGrid=gs.R_GRID):
    """Exhaustive sweep, k analytic.  Same objective, same cache."""
    eta = obj.eta
    rows = []
    for x in xList:
        Q = obj.c['Q'][x]
        HR = obj.c['HR']
        n = len(HR)
        for a in alphaGrid:
            M = obj.M(a)
            P = {q: Q[q] @ M.T for q in obj.QoIs}
            k, k_raw = optimal_k(P['u'], obj.T['u'], *obj.kBounds)
            blk = {'$h$': HR[:, 0], '$r$': HR[:, 1], '$x$': np.full(n, x),
                   r'$\alpha$': np.full(n, a), r'$k$': k, 'k_unclipped': k_raw}
            for q in obj.QoIs:
                pred = k[:, None] * P[q] if q == 'u' else P[q]
                blk[q] = np.linalg.norm(pred - obj.T[q], axis=1)
            rows.append(pd.DataFrame(blk))
    return pd.concat(rows, ignore_index=True)
