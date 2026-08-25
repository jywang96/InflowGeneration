"""
Exhaustive grid search over the *original* point-wise GPR models.

Purpose: run the same objective NSGA-II minimises, but over the complete
discrete design space instead of a stochastic sample, so the two can be
compared head to head on identical surrogates.

Why the space is enumerable at all: MyProblem.eval_model_delayed discretises
its own decision variables on entry --

    h -> np.round(h*100)/100      over [0.035, 0.1649]  ->  13 values
    r -> np.round(r)              over [51.5,  92.49 ]  ->  41 values
    x -> xList[int(np.round(.))]  over [-0.49, 17.49 ]  ->  18 values

-- so NSGA-II is already searching a 13*41*18 = 9594 point grid (times the
continuous alpha and k).  k is eliminated in closed form (see common.optimal_k)
and alpha is a 1-D scan, which leaves a sweep that is cheaper in GPR calls than
the 28160 evaluations NSGA-II spends: the grid needs one prediction per
(h, r, x), whereas NSGA-II re-predicts the same configuration for every new
(alpha, k) pair it tries.

Two deviations from modelDefinition, both for tractability, both verified:

  * Models are loaded one (x, QoI) at a time instead of all 76 at once.  Each
    pickle is ~72 MB (a 3000x3000 Cholesky factor), so MyProblem's preload
    strategy would need ~5.5 GB resident.
  * predict_mean() asks sklearn for the posterior mean only.  gaussianProcess
    .predict hardcodes return_std=True, and the variance costs
    O(n_train^2 * n_query) -- hours here, for a quantity the objective never
    reads.  check_predict_mean() asserts the means agree exactly.
"""

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelDefinition import gaussianProcess                      # noqa: E402
from hyperparametersGPR import trainPairs, devPairs, testPairs   # noqa: E402

from common import Y_GRID, FEATURES, alpha_operator, optimal_k   # noqa: E402

# The x stations optimizeParameters.py exposes to NSGA-II (note: no 0.3).
X_LIST = [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0,
          3.3, 3.6, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0]

# The reachable values after eval_model_delayed's rounding.
H_GRID = np.round(np.arange(0.04, 0.1601, 0.01), 2)
R_GRID = np.arange(52, 93, dtype=float)

QOIS = ('u', 'Iu', 'Iv', 'Iw')
MODELDIR = '../GPRModels'
DATABASE = './GPRDatabase'


def station(x, yMax=1.0, database=DATABASE):
    """gaussianProcess for one x, carrying that station's standardisation.

    No model is loaded yet -- attach one with load_qoi().  The per-x meanVal /
    stdVal come from that station's training data, so this cannot be shared
    across x.
    """
    return gaussianProcess(
        {'h': trainPairs[:, 0], 'r': trainPairs[:, 1], 'x': [x]},
        {'h': devPairs[:, 0],   'r': devPairs[:, 1],   'x': [x]},
        {'h': testPairs[:, 0],  'r': testPairs[:, 1],  'x': [x]},
        yMax, database)


def load_qoi(gp, x, QoI, testID='intensities', modeldir=MODELDIR):
    """Attach the (x, QoI) pickle, releasing whichever was attached before."""
    gp.predictive_model = None
    gc.collect()
    prefix = str(str(x) + '_').replace('.', 'p')
    gp.loadModel(os.path.join(modeldir, f'{prefix}{testID}_{QoI}.pkl'))
    return gp


def _standardise(gp, df):
    return ((df[list(FEATURES)] - gp.meanVal[list(FEATURES)])
            / gp.stdVal[list(FEATURES)]).to_numpy()


def predict_mean(gp, hr, yGrid=Y_GRID, chunk=25):
    """Posterior mean for every (h, r) in hr, on yGrid.  Shape (len(hr), n_y)."""
    hr = np.atleast_2d(np.asarray(hr, dtype=float))
    ny = len(yGrid)
    out = np.empty((len(hr), ny))

    for i0 in range(0, len(hr), chunk):
        blk = hr[i0:i0 + chunk]
        df = pd.DataFrame({
            'y': np.tile(yGrid, len(blk)),
            'h': np.repeat(blk[:, 0], ny),
            'r': np.repeat(blk[:, 1], ny),
        })
        mu = gp.predictive_model.predict(_standardise(gp, df), return_std=False)
        out[i0:i0 + len(blk)] = np.asarray(mu).ravel().reshape(len(blk), ny)

    return out


def check_predict_mean(gp, hr=((0.08, 72.0),), atol=1e-12):
    """Assert predict_mean matches gaussianProcess.predict's y_model column."""
    hr = np.atleast_2d(np.asarray(hr, dtype=float))
    fast = predict_mean(gp, hr)
    worst = 0.0
    for i, (h, r) in enumerate(hr):
        df = pd.DataFrame({'y': Y_GRID, 'h': h, 'r': r})
        df = gp.predict(None, df, list(FEATURES), 'unused')
        worst = max(worst, float(np.max(np.abs(df['y_model'].to_numpy() - fast[i]))))
    if worst > atol:
        raise AssertionError(f'predict_mean deviates by {worst:.3e}')
    return worst


def _profiles_for_x(x, QoIs=QOIS, hGrid=H_GRID, rGrid=R_GRID, testID='intensities',
                    yMax=1.0, database=DATABASE, modeldir=MODELDIR, verify=False):
    """(HR, {QoI: (n_hr, n_y) array}) for one station, one model resident."""
    HR = np.array([(h, r) for h in hGrid for r in rGrid], dtype=float)
    gp = station(x, yMax, database)
    Q = {}
    for QoI in QoIs:
        load_qoi(gp, x, QoI, testID, modeldir)
        if verify:
            check_predict_mean(gp)
            verify = False
        Q[QoI] = predict_mean(gp, HR)
    gp.predictive_model = None
    del gp
    gc.collect()
    return HR, Q


def reevaluate(target, QoIs, decisions, xList=X_LIST, yGrid=Y_GRID, **kw):
    """Score explicit (h, r, alpha, k, x) rows with the same objective.

    decisions : DataFrame with columns '$h$','$r$',r'$\\alpha$',r'$k$','$x$',
                where '$x$' is the *raw* NSGA-II gene (a float index into
                xList), exactly as written to *_decision_variables.csv.

    Use this to confirm the committed *_pareto_front.csv can be reproduced
    before drawing any conclusion from a dominance comparison -- if it cannot,
    the two runs are not using the same objective and are not comparable.
    """
    eta = target['y'].to_numpy()
    T = {q: target[q].to_numpy() for q in QoIs}

    dec = decisions.reset_index(drop=True).copy()
    dec['_h'] = np.round(dec['$h$'] * 100) / 100
    dec['_r'] = np.round(dec['$r$'])
    dec['_x'] = [xList[int(np.round(v))] for v in dec['$x$']]

    out = np.full((len(dec), len(QoIs)), np.nan)
    for x, sub in dec.groupby('_x'):
        pos = sub.index.to_numpy()
        hr = sub[['_h', '_r']].to_numpy()
        Ms = [alpha_operator(a, eta, yGrid) for a in sub[r'$\alpha$']]
        ks = sub[r'$k$'].to_numpy()

        gp = station(x, kw.get('yMax', 1.0), kw.get('database', DATABASE))
        for j, q in enumerate(QoIs):
            load_qoi(gp, x, q, kw.get('testID', 'intensities'),
                     kw.get('modeldir', MODELDIR))
            Q = predict_mean(gp, hr, yGrid)
            for i in range(len(sub)):
                pred = Ms[i] @ Q[i]
                if q == 'u':
                    pred = pred * ks[i]
                out[pos[i], j] = np.linalg.norm(pred - T[q])
        gp.predictive_model = None
        del gp
        gc.collect()

    return pd.DataFrame(out, columns=list(QoIs), index=decisions.index)


def sweep(target, QoIs, alphaGrid, hGrid=H_GRID, rGrid=R_GRID, xList=X_LIST,
          kBounds=(0.8, 1.8), verbose=True, **kw):
    """Full (h, r, x) x alpha sweep with k solved analytically.

    target : normalised target ABL DataFrame from common.load_target
    Returns one row per (h, r, x, alpha).
    """
    eta = target['y'].to_numpy()
    T = {q: target[q].to_numpy() for q in QoIs}

    # alpha enters only through this operator, which does not depend on
    # (h, r, x) -- build it once per alpha and reuse over the whole grid.
    Malpha = {a: alpha_operator(a, eta) for a in alphaGrid}

    rows = []
    for xi, x in enumerate(xList):
        t0 = time.time()
        HR, Q = _profiles_for_x(x, QoIs, hGrid, rGrid, verify=(xi == 0), **kw)
        n = len(HR)

        for a in alphaGrid:
            M = Malpha[a]
            P = {q: Q[q] @ M.T for q in QoIs}                    # (n, n_t)

            if 'u' in QoIs:
                k, k_raw = optimal_k(P['u'], T['u'], *kBounds)
            else:
                k = k_raw = np.ones(n)

            blk = {'$h$': HR[:, 0], '$r$': HR[:, 1], '$x$': np.full(n, x),
                   r'$\alpha$': np.full(n, a), r'$k$': k, 'k_unclipped': k_raw}
            for q in QoIs:
                pred = k[:, None] * P[q] if q == 'u' else P[q]
                blk[q] = np.linalg.norm(pred - T[q], axis=1)
            rows.append(pd.DataFrame(blk))

        if verbose:
            print(f'  x={x:<5}  {n * len(alphaGrid):>7d} points  '
                  f'({time.time() - t0:.1f}s)')

    return pd.concat(rows, ignore_index=True)
