"""
Shared helpers for the dev/ experiments.

Everything here is written to reproduce the *exact* objective function used by
optimizeParameters.py / MyProblem.eval_model_delayed, but vectorised so that a
full grid can be swept instead of sampled by NSGA-II.

The fidelity of the vectorised path is checked against the original
modelDefinition.scale_predictions in check_alpha_operator().
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelDefinition import scale_predictions  # noqa: E402

# --------------------------------------------------------------------------
# The y grid MyProblem evaluates the GPR on.  The original concatenation
# repeats y = 0.3 (last point of the first linspace, first of the second);
# np.unique drops the duplicate, which changes nothing numerically -- the GPR
# returns the same value twice -- but keeps the interpolation well posed.
# --------------------------------------------------------------------------
Y_GRID = np.unique(
    np.concatenate((np.linspace(0.01, 0.3, 200), np.linspace(0.3, 1.0, 200)))
)

FEATURES = ['y', 'h', 'r']


def load_target(fName, testcases='./TestCases'):
    """Read and normalise a target ABL exactly as optimizeParameters.py does."""
    ref = pd.read_csv(os.path.join(testcases, fName + '.dat'), sep=',')
    header = list(ref.columns)
    idx = int(np.argmax(ref['y'].to_numpy()))

    Uref = 1.0 * ref['u'].iloc[idx]
    yref = 1.0 * ref['y'].iloc[idx]

    ref['y'] = ref['y'] / yref
    ref['u'] = ref['u'] / Uref
    for rsc in ['uu', 'vv', 'ww', 'uv', 'uw', 'vw']:
        if rsc in header:
            ref[rsc] = ref[rsc] / (Uref ** 2)

    return ref, header, Uref, yref


def interp_matrix(src_y, query_y):
    """Linear-interpolation matrix M with  M @ f  ==  np.interp(query_y, src_y, f).

    src_y must be strictly increasing and must bracket query_y.
    """
    src_y = np.asarray(src_y, dtype=float)
    query_y = np.asarray(query_y, dtype=float)

    if np.any(query_y < src_y[0] - 1e-12) or np.any(query_y > src_y[-1] + 1e-12):
        raise ValueError('query_y falls outside src_y; extrapolation not allowed')

    idx = np.clip(np.searchsorted(src_y, query_y) - 1, 0, len(src_y) - 2)
    w = (query_y - src_y[idx]) / (src_y[idx + 1] - src_y[idx])

    M = np.zeros((len(query_y), len(src_y)))
    rows = np.arange(len(query_y))
    M[rows, idx] = 1.0 - w
    M[rows, idx + 1] = w
    return M


def alpha_operator(alpha, eta_target, y_grid=Y_GRID):
    """Vectorised form of scale_predictions().

    scale_predictions truncates the model profile at y <= alpha*max(y), rescales
    the surviving y so its maximum is 1, and interpolates onto the target's y.
    That is algebraically identical to evaluating the *untruncated* profile at
    y = y_trunc_max * eta_target, where y_trunc_max is the largest grid point
    still <= alpha.

    Returns M with shape (len(eta_target), len(y_grid)) so that  M @ q  is the
    rescaled profile sampled on the target grid.
    """
    mask = y_grid <= alpha * y_grid.max()
    if mask.sum() < 2:
        raise ValueError(f'alpha={alpha} truncates the profile to <2 points')
    y_trunc_max = y_grid[mask].max()

    M_trunc = interp_matrix(y_grid[mask], y_trunc_max * eta_target)

    M = np.zeros((len(eta_target), len(y_grid)))
    M[:, mask] = M_trunc
    return M


def check_alpha_operator(profile, eta_target, alphas, atol=1e-10):
    """Assert alpha_operator reproduces scale_predictions on a sample profile.

    profile : 1-D array of GPR predictions on Y_GRID
    Returns the largest absolute discrepancy found.
    """
    target_df = pd.DataFrame({'y': eta_target, 'u': np.zeros_like(eta_target)})
    worst = 0.0
    for alpha in alphas:
        model_df = pd.DataFrame({'y': Y_GRID.copy(), 'y_model': profile.copy()})
        _, ref = scale_predictions(model_df, target_df, alpha, 'u')
        fast = alpha_operator(alpha, eta_target) @ profile
        worst = max(worst, float(np.max(np.abs(ref - fast))))
    if worst > atol:
        raise AssertionError(
            f'alpha_operator deviates from scale_predictions by {worst:.3e}')
    return worst


def anchor_index(eta, y_ref, y_T):
    """Index of the target point closest to the reference height.

    y_ref is given in the same units as the target's y column; y_T is the top
    of the target profile, so the normalised anchor is eta_ref = y_ref / y_T.
    Asking the user for a physical height rather than a ratio avoids the trap
    that eta_ref depends on how far above the building the target was
    specified: the same 30 m building gives eta_ref = 0.667 for a target
    reaching 1.5H and 0.5 for one reaching 2H.
    """
    eta_ref = float(y_ref) / float(y_T)
    if not (eta.min() - 1e-9 <= eta_ref <= eta.max() + 1e-9):
        raise ValueError(
            f'reference height y_ref={y_ref} maps to eta_ref={eta_ref:.4f}, '
            f'outside the target range [{eta.min():.4f}, {eta.max():.4f}]')
    return int(np.argmin(np.abs(eta - eta_ref))), eta_ref


def optimal_k(P, t, kmin=0.8, kmax=1.8):
    """Least-squares velocity scale: the value minimising ||s_U p - t||.

    P : (N, n_t) candidate u profiles on the target grid
    t : (n_t,)   target u profile

    Retained for diagnostics.  It is the smallest attainable velocity
    residual, so the gap to the anchored scale measures what anchoring costs.
    """
    num = P @ t
    den = np.einsum('ij,ij->i', P, P)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.where(den > 0, num / den, kmin)
    return np.clip(k, kmin, kmax), k


def anchored_k(P, t, j, kmin=0.8, kmax=1.8):
    """Velocity scale that makes candidate and target agree at index j.

    Wind loading is reported as a coefficient normalised by the dynamic
    pressure at the building height, so an error there enters the deliverable
    twice over: C_p scales as U_ref^-2.  Fixing the amplitude at that height
    removes a systematic bias from the quantity the simulation exists to
    produce, at the cost of a slightly larger whole-profile residual.

    Averaging the anchor over a window of neighbouring heights was tried and
    abandoned: the surrogate error is smooth in y -- adjacent grid points
    correlate at r = 0.996 -- so averaging correlated values reduces it by
    0.4%, while blurring the property being enforced.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.where(np.abs(P[:, j]) > 0, t[j] / P[:, j], kmin)
    return np.clip(k, kmin, kmax), k


def velocity_scale(P, t, mode='anchor', j=None, kmin=0.8, kmax=1.8):
    """Dispatch: 'anchor' (delivered) or 'ls' (diagnostic)."""
    if mode == 'anchor':
        if j is None:
            raise ValueError("mode='anchor' needs the anchor index j")
        return anchored_k(P, t, j, kmin, kmax)
    if mode == 'ls':
        return optimal_k(P, t, kmin, kmax)
    raise ValueError(f'unknown mode {mode!r}')


def dominates(A, f):
    """Rows of A that dominate f (minimisation, weak dominance + one strict)."""
    return np.all(A <= f, axis=1) & np.any(A < f, axis=1)


def pareto_front(F):
    """Indices of the non-dominated rows of F (minimisation).

    Sorted-sweep with a small archive; fine for the ~1e4-1e5 point sets we keep
    after per-configuration filtering, too slow for the raw 1e6 grid.
    """
    order = np.argsort(F.sum(axis=1))
    Fs = F[order]
    keep = []
    for i in range(len(Fs)):
        if keep and np.any(dominates(Fs[keep], Fs[i])):
            continue
        keep.append(i)
    return order[np.asarray(keep, dtype=int)]
