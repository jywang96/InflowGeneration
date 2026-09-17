"""Build a runnable downstream case from an optimiser solution.

Shared by `12_make_cases.py` (the ASCE cases) and `13_wind_tunnel_optima.py`
(the two tunnel cases), so the two cannot drift apart.

**Why this exists rather than `generateInflow.py`.**  That script loads
`../GPRModels/{-4p95,-2p85}_inflow_stresses_*.pkl`: the superseded point-wise
models, and a separate set per plane.  Paper section 4.3.1 describes a single
upstream surrogate pooled by true row index `n`, which is also the surrogate
Figure 4 validates.  Boundary conditions written the old way would verify a
framework the paper does not describe.  See HANDOVER 22.

The two factors that are easy to confuse:

  s_y = y_T / y^Max      the similarity ratio of paper eq (eqn:sy): how far the
                         model-scale flow must be stretched to become the
                         target.
  geo                    the geometric factor of the scaling study.  It is what
                         `modelDefinition` multiplies every domain dimension
                         by, and `generateInflow` computes it as
                         1.5*H_build*scale/y^Max = s_y * scale, so the
                         `caseConfig` entry is scale = geo / s_y.  geo = 1 is
                         the setup the optimiser selected; geo = 6 is that
                         setup enlarged sixfold.  The building scale is then
                         1 : s_y/geo.

`H_build` must be the building height **in the units of the target file**.  The
ASCE targets are full scale, so it is 6, 30 or 100 m; the two tunnel targets
are at model scale, so it is y_T/1.5 there and the full-scale height is
context, not an input.
"""

import json
import os

import numpy as np
import pandas as pd

import podlib as pl
from hyperparametersGPR import trainPairs

QU = ['u', 'uu', 'vv', 'ww', 'uv']
# generateInflow.py writes the BCs on this grid; keep it identical so the
# server-side deck needs no change.
N_BC = 1501
Y_BC_MAX = 1.0
# loadData takes a *normalised* grid, y / yMax, with yMax the physical top.
# The upstream surrogate is fitted over y in [0.015, 1.5] m (paper section 6),
# so Y_FIT is eta = y / Y_TOP.
Y_TOP = 1.5
Y_FIT = np.linspace(0.01, 1.0, 400)
ROW_OFFSET = pl.ROW_OFFSET

_SU = None


def upstream_surrogate(verbose=True):
    """Fit (once) the pooled upstream surrogate S^U over (h, n)."""
    global _SU
    if _SU is not None:
        return _SU
    if verbose:
        print('fitting S^U (pooled by row index n)')
    meta, X = pl.build_upstream_snapshots(trainPairs, QU, Y_FIT, Y_TOP)
    out = {}
    for q in QU:
        s = pl.PODSurrogate(n_modes=5, features=('h', 'n'), log_cols=()
                            ).fit(meta, X[q], Y_FIT)
        out[q] = s
        if verbose:
            print(f'  {q:<3} M={s.n_modes_used} E_M={s.E_M():.6f} '
                  f'e_trunc={s.truncation_error(X[q]):.3e}'
                  + (f'  dropped {s.dropped}' if s.dropped else ''))
    if verbose:
        print('  training stations n: '
              f'{sorted(meta["n"].unique().astype(int).tolist())}')
    _SU = (out, meta, X)
    return _SU


def upstream_bc(h, n):
    """Dimensionless upstream profiles at (h, n), on Y_FIT."""
    su, _, _ = upstream_surrogate()
    g = pd.DataFrame({'h': [float(h)], 'n': [float(n)]})
    return {q: su[q].predict(g)[0] for q in QU}


def write_case(outdir, name, fname, h, r, x_B, y_Max, s_U, H_build,
               geo=1.0, U_yT=None, y_T=None, dimensional=True, notes=()):
    """One case directory: two BC tables plus caseConfig.json.

    `U_yT` and `y_T` are read from the target file unless given.  Set
    `dimensional=False` when the target's velocity column is normalised, in
    which case `Vinlet` is dimensionless and the case cannot be run until a
    reference wind speed is supplied -- the config records that explicitly
    rather than writing a meaningless number.
    """
    if y_T is None or U_yT is None:
        tgt = pd.read_csv(f'TestCases/{fname}.dat')
        i = int(np.argmax(tgt['y'].to_numpy()))
        y_T = float(tgt['y'].iloc[i])
        U_yT = float(tgt['u'].iloc[i])

    Vinlet = s_U * U_yT
    s_y = y_T / y_Max
    scale_cfg = geo / s_y
    y_bc = np.linspace(0.01, Y_BC_MAX, N_BC) * geo

    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)

    for plane, n, lab in ((-4.95, r - ROW_OFFSET, 'inflow_input'),
                          (-2.85, r, 'ALF_input')):
        p = upstream_bc(h, n)
        # Both grids are the same normalised coordinate, so this is a
        # resampling and carries no scale factor.  Geometric similarity is
        # what lets one eta-profile serve every geo: only y_bc is stretched.
        eta_bc = np.linspace(0.01, Y_BC_MAX, N_BC)
        pi = {q: np.interp(eta_bc, Y_FIT, p[q]) for q in QU}
        pd.DataFrame({
            'x': plane, 'y': y_bc, 'z': 0.0,
            'x-velocity': pi['u'] * Vinlet,
            'y-velocity': 0.0, 'z-velocity': 0.0,
            'uu-reynolds-stress': np.abs(pi['uu']) * Vinlet ** 2,
            'vv-reynolds-stress': np.abs(pi['vv']) * Vinlet ** 2,
            'ww-reynolds-stress': np.abs(pi['ww']) * Vinlet ** 2,
            'uv-reynolds-stress': -np.abs(pi['uv']) * Vinlet ** 2,
            'uw-reynolds-stress': 0.0, 'vw-reynolds-stress': 0.0,
        }).to_csv(f'{d}/{fname}_{lab}.txt', sep='\t', index=False)

    cfg = {
        '_comment': [
            'Generated by dev/caselib.py via dev/12_make_cases.py or',
            'dev/13_wind_tunnel_optima.py.  The boundary conditions come from',
            'the POD+GPR upstream surrogate pooled by row index n, NOT from',
            'the point-wise pickles generateInflow.py reads.  HANDOVER 22.',
            'alpha is y^Max; k is the velocity scale s_U.',
            'y_ref is a height in the units of the target y column.',
        ] + list(notes),
        'reference': {'fName': fname, 'h': float(h), 'r': int(r),
                      'alpha': float(y_Max), 'k': float(s_U),
                      'x': float(x_B), 'y_ref': float(H_build)},
        'scaleFactors': {'scale': float(scale_cfg),
                         'H_build': float(H_build)},
        'wDomain': 3.0, 'plotABL': False,
        '_derived': {
            'y_T': y_T, 'U_yT': U_yT, 'Vinlet': Vinlet, 's_y': s_y,
            'geometric_factor': geo, 'building_scale': f'1:{s_y / geo:.1f}',
            'n_inlet': int(r - ROW_OFFSET), 'n_alf': int(r),
            'velocity_is_dimensional': bool(dimensional),
        },
    }
    if not dimensional:
        cfg['_derived']['BLOCKED'] = (
            "the target's velocity column is normalised, so Vinlet above is "
            'dimensionless.  Multiply every velocity by the tunnel reference '
            'wind speed at y_ref, and every Reynolds stress by its square, '
            'before running.  See HANDOVER 23.')
    with open(f'{d}/caseConfig.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg['_derived']
