"""POD + GPR surrogate for the downstream ABL database.

The shipped surrogate treats every (y, h, r) triple as an independent scalar
observation: 20 profiles x 100 heights = 2000 training points per (x, QoI), a
2000x2000 Cholesky factor (72 MB on disk), and 19*4 = 76 separate models.  Two
consequences follow from putting `y` in the feature vector alongside `h` and
`r`:

  * The kernels are isotropic.  `RBF()` with no arguments is a *scalar*
    length_scale in scikit-learn, so after z-scoring the model is forced to use
    one correlation length for the profile's smoothness in y, its sensitivity
    to roughness height, and its sensitivity to fetch.  Those are unrelated
    quantities.  ABL profiles are also strongly non-stationary in y -- a sharp
    canopy-wake inflection near the ground, a plateau aloft -- which no single
    length scale can represent.
  * `x` cannot be a feature (2000*19 points would be intractable), so each
    station gets its own model and the optimiser sees `x` as an integer index.

POD separates the two roles that were conflated: `y` is a coordinate *within* a
sample, `h, r, x` are parameters *between* samples.  Expand each profile on a
basis learned from the data,

    q(y; h,r,x)  ~  qbar(y) + sum_k a_k(h,r,x) phi_k(y)

and regress only the coefficients.  The training set becomes 20*19 = 380
vector-valued samples over a 3-D input, `x` becomes continuous, predictions are
confined to the span of profiles the LES actually produced, and the kernel is
free to be anisotropic over (h, r, x) alone.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelDefinition import loadData                              # noqa: E402

from sklearn.gaussian_process import GaussianProcessRegressor     # noqa: E402
from sklearn.gaussian_process.kernels import (                    # noqa: E402
    ConstantKernel, Matern, WhiteKernel)

DATABASE = './GPRDatabase'
REF_VELOCITY = 15.0


def build_snapshots(pairs, xList, QoIs, yGrid, yMax=1.0, database=DATABASE,
                    refvel=REF_VELOCITY):
    """Assemble the snapshot matrices.

    One loadData call per (h, r) covers every station, so each PFh*.txt is read
    once instead of once per (h, r, x).

    Returns (meta, X) where meta has columns ['h','r','x'] with one row per
    profile and X maps each QoI to an array of shape (len(meta), len(yGrid)).
    """
    QoIs = list(QoIs)
    rows, blocks = [], {q: [] for q in QoIs}

    for h, r in np.asarray(pairs, dtype=float):
        df = loadData([h], list(xList), [r], yMax, database, refvel, yGrid)
        for x in xList:
            sub = df[np.abs(df['x'] - x) < 1e-9].sort_values('y')
            if len(sub) != len(yGrid):
                raise RuntimeError(
                    f'h={h} r={r} x={x}: got {len(sub)} rows, want {len(yGrid)}')
            rows.append({'h': h, 'r': r, 'x': x})
            for q in QoIs:
                blocks[q].append(sub[q].to_numpy())

    meta = pd.DataFrame(rows)
    X = {q: np.asarray(blocks[q]) for q in QoIs}
    return meta, X


def pod(X, n_modes=None):
    """Mean-centred POD of a snapshot matrix.

    X : (n_samples, n_y)
    Returns (qbar, Phi, A, energy) with Phi (n_y, n_modes), A (n_samples,
    n_modes) and `energy` the cumulative fraction of variance over *all* modes.
    """
    qbar = X.mean(axis=0)
    Xc = X - qbar
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)

    k = len(S) if n_modes is None else int(n_modes)
    Phi = Vt[:k].T
    A = Xc @ Phi
    return qbar, Phi, A, energy


def modes_needed(energy, tol=1e-3):
    """Smallest mode count capturing 1 - tol of the variance."""
    return int(np.searchsorted(energy, 1.0 - tol) + 1)


class PODSurrogate:
    """(h, r, x) -> full profile, via GPR on POD coefficients.

    One GPR per retained mode.  Each gets its own ARD Matern kernel, so the
    correlation lengths in h, r and x are fitted independently -- the freedom
    the shipped isotropic kernels do not have.  With ~380 samples the fits take
    seconds, so the hyperparameter search can be a proper restart-based MLL
    optimisation rather than a 256-seed random draw.
    """

    def __init__(self, n_modes=5, log_x=True, n_restarts=6, random_state=0):
        self.n_modes = n_modes
        self.log_x = log_x
        self.n_restarts = n_restarts
        self.random_state = random_state

    # -- input handling ----------------------------------------------------
    def _raw(self, meta):
        Z = np.column_stack([
            np.asarray(meta['h'], dtype=float),
            np.asarray(meta['r'], dtype=float),
            np.log(np.asarray(meta['x'], dtype=float)) if self.log_x
            else np.asarray(meta['x'], dtype=float),
        ])
        return Z

    def _scale(self, Z):
        return (Z - self._zmean) / self._zstd

    # -- fit / predict -----------------------------------------------------
    def fit(self, meta, X, yGrid):
        self.yGrid = np.asarray(yGrid, dtype=float)

        Z = self._raw(meta)
        self._zmean, self._zstd = Z.mean(axis=0), Z.std(axis=0)
        Zs = self._scale(Z)

        self.qbar, self.Phi, A, self.energy = pod(X, self.n_modes)

        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(Zs.shape[1]),
                           length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(1e-6, (1e-12, 1e-1)))

        self.gprs = []
        for k in range(self.Phi.shape[1]):
            g = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                n_restarts_optimizer=self.n_restarts,
                random_state=self.random_state)
            g.fit(Zs, A[:, k])
            self.gprs.append(g)
        return self

    def coefficients(self, meta):
        Zs = self._scale(self._raw(meta))
        return np.column_stack([g.predict(Zs) for g in self.gprs])

    def predict(self, meta):
        """Profiles on self.yGrid, shape (len(meta), len(yGrid))."""
        return self.qbar + self.coefficients(meta) @ self.Phi.T

    # -- diagnostics -------------------------------------------------------
    def truncation_error(self, X):
        """RMSE of POD reconstruction alone, with no regression involved.

        This is the accuracy ceiling: no amount of regression skill can beat
        it, so check it before tuning anything else.
        """
        Xhat = self.qbar + (X - self.qbar) @ self.Phi @ self.Phi.T
        return float(np.sqrt(np.mean((X - Xhat) ** 2)))

    def total_error(self, meta, X):
        """RMSE of the full surrogate (truncation + regression)."""
        return float(np.sqrt(np.mean((X - self.predict(meta)) ** 2)))

    def length_scales(self):
        """Fitted ARD length scales per mode, as a tidy DataFrame."""
        names = ['h', 'log x' if self.log_x else 'x']
        names.insert(1, 'r')
        out = []
        for k, g in enumerate(self.gprs):
            ls = g.kernel_.k1.k2.length_scale
            out.append(dict(zip(names, np.atleast_1d(ls))) | {'mode': k + 1})
        return pd.DataFrame(out).set_index('mode')
