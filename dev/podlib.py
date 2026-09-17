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

# --------------------------------------------------------------------------
# Upstream stations
#
# The database labels the two upstream planes with pseudo-coordinates that
# locate them in the *downstream* domain's frame, whose origin is the end of
# the roughness canopy (upstreamLength = 2.1 + 0.15 + 2.7 = 4.95 m):
#
#   x = -4.95  ->  downstream inlet (domain x = 0),      feeds INLET_PROFILE
#   x = -2.85  ->  end of the ALF box (domain x = 2.1),  feeds DATA_ALF
#
# In the upstream canopy the two planes are the same quantity sampled 7 rows
# (2.1 m) apart, so they belong in one model indexed by the true row number.
# The label r names the ALF station; the inlet station is 7 rows upstream:
#
#   n(-2.85) = r        n(-4.95) = r - 7
#
# Consistency check against formatSetup's lookup tables, which place the ALF
# station of label 52 at 23.65 m and the inlet station at 21.55 m: the two
# differ by 2.1 m = 7 rows at the 0.3 m row spacing, and consecutive labels
# differ by 5 rows = 1.5 m, both of which the relation above reproduces.
# --------------------------------------------------------------------------
X_INLET, X_ALF = -4.95, -2.85
ROW_OFFSET = 7


def row_index(r, x):
    """True upstream row index for a database entry labelled (r, x)."""
    if np.isclose(x, X_ALF):
        return float(r)
    if np.isclose(x, X_INLET):
        return float(r) - ROW_OFFSET
    raise ValueError(f'x={x} is not an upstream station')


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


def build_upstream_snapshots(pairs, QoIs, yGrid, yMax=1.5, database=DATABASE,
                             refvel=REF_VELOCITY):
    """Snapshots for the upstream database, indexed by true row number.

    Both upstream planes of every (h, r) file are pooled, so each h
    contributes twice as many profiles as the number of r labels: for
    h in {0.04, 0.08, 0.12, 0.16} the labels {52,62,72,82,92} yield rows
    {45,52,55,62,65,72,75,82,85,92}, ten distinct stations instead of five.

    Returns (meta, X) with meta columns ['h', 'n', 'r', 'plane'], where
    'plane' records which boundary condition the profile feeds.
    """
    QoIs = list(QoIs)
    rows, blocks = [], {q: [] for q in QoIs}

    for h, r in np.asarray(pairs, dtype=float):
        df = loadData([h], [X_INLET, X_ALF], [r], yMax, database, refvel, yGrid)
        for x in (X_INLET, X_ALF):
            sub = df[np.abs(df['x'] - x) < 1e-9].sort_values('y')
            if len(sub) != len(yGrid):
                raise RuntimeError(
                    f'h={h} r={r} x={x}: got {len(sub)} rows, want {len(yGrid)}')
            rows.append({'h': h, 'n': row_index(r, x), 'r': r,
                         'plane': 'inlet' if x == X_INLET else 'alf'})
            for q in QoIs:
                blocks[q].append(sub[q].to_numpy())

    meta = pd.DataFrame(rows).sort_values(['h', 'n']).reset_index()
    order = meta.pop('index').to_numpy()
    X = {q: np.asarray(blocks[q])[order] for q in QoIs}
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

    LS_BOUNDS = (1e-2, 1e3)

    def __init__(self, n_modes=5, features=('h', 'r', 'x'), log_cols=('x',),
                 n_restarts=6, random_state=0, drop_degenerate=True,
                 ls_margin=0.05):
        """features : meta columns used as GP inputs.

        Downstream surrogate: ('h', 'r', 'x') with x on a log scale, since the
        sampling stations are logarithmically spaced.  Upstream surrogate:
        ('h', 'n') with no log, the row index being uniformly spaced.

        drop_degenerate : truncate at the first mode whose fitted ARD length
            scale reaches a bound of `LS_BOUNDS`.  A length scale at its lower
            bound means the process has stopped interpolating in that input
            and is fitting the mode as noise; retaining it adds a
            noise-driven term to the reconstruction.  `n_modes` is therefore
            a ceiling rather than an exact count.

            Truncation is a *prefix*: the first offending mode and everything
            after it are dropped.  Dropping an interior mode would leave the
            retained set no longer the leading-M subspace, which is what the
            Eckart-Young-Mirsky optimality of the basis rests on.

        ls_margin : a length scale counts as degenerate when it sits within
            this fraction of the *logarithmic* width of `LS_BOUNDS` of either
            end.  The bounds span five decades and scikit-learn optimises the
            log length scale, so proximity has to be measured in decades: on
            the default bounds a fitted 0.013 is 2.3 per cent of the way up
            from the lower bound, which a plain multiplicative test would call
            30 per cent clear of it and miss.
        """
        self.n_modes = n_modes
        self.features = tuple(features)
        self.log_cols = tuple(c for c in log_cols if c in self.features)
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.drop_degenerate = drop_degenerate
        self.ls_margin = ls_margin

    # -- input handling ----------------------------------------------------
    def _raw(self, meta):
        cols = []
        for c in self.features:
            v = np.asarray(meta[c], dtype=float)
            cols.append(np.log(v) if c in self.log_cols else v)
        return np.column_stack(cols)

    def _feature_names(self):
        return [f'log {c}' if c in self.log_cols else c for c in self.features]

    def _scale(self, Z):
        return (Z - self._zmean) / self._zstd

    # -- fit / predict -----------------------------------------------------
    def fit(self, meta, X, yGrid):
        self.yGrid = np.asarray(yGrid, dtype=float)

        Z = self._raw(meta)
        self._zmean, self._zstd = Z.mean(axis=0), Z.std(axis=0)
        Zs = self._scale(Z)

        self.qbar, self.Phi, A, self.energy = pod(X, self.n_modes)

        lo, hi = self.LS_BOUNDS
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(Zs.shape[1]),
                           length_scale_bounds=self.LS_BOUNDS, nu=2.5)
                  + WhiteKernel(1e-6, (1e-12, 1e-1)))

        self.gprs = []
        self.dropped = []
        for k in range(self.Phi.shape[1]):
            g = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                n_restarts_optimizer=self.n_restarts,
                random_state=self.random_state)
            g.fit(Zs, A[:, k])

            ls = np.atleast_1d(g.kernel_.k1.k2.length_scale)
            frac = (np.log10(ls) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            at_bound = (frac <= self.ls_margin) | (frac >= 1.0 - self.ls_margin)
            if self.drop_degenerate and at_bound.any():
                names = self._feature_names()
                self.dropped = [
                    (k + 1, [(names[i], float(ls[i]))
                             for i in np.flatnonzero(at_bound)])]
                break
            self.gprs.append(g)

        # Prefix truncation: keep the basis and the regressions the same size.
        self.n_modes_used = len(self.gprs)
        if self.n_modes_used == 0:
            raise RuntimeError(
                'every mode has a length scale at a bound; the inputs are '
                'probably not informative for this QoI')
        self.Phi = self.Phi[:, :self.n_modes_used]
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

    def E_M(self):
        """Retained variance fraction at the number of modes actually used."""
        return float(self.energy[self.n_modes_used - 1])

    def length_scales(self):
        """Fitted ARD length scales per mode, as a tidy DataFrame."""
        names = self._feature_names()
        out = []
        for k, g in enumerate(self.gprs):
            ls = np.atleast_1d(g.kernel_.k1.k2.length_scale)
            out.append(dict(zip(names, ls)) | {'mode': k + 1})
        return pd.DataFrame(out).set_index('mode')
