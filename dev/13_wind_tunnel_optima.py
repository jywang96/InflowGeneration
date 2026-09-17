# %% [markdown]
# # Group D of HANDOVER 21: the two wind-tunnel targets, under the POD surrogate
#
# Paper section 7.2 claims the framework reproduces boundary layers from
# facilities it was not built to imitate.  Those optima were computed with the
# superseded point-wise surrogate, like the ASCE nine, so they are re-run here.
#
# The two targets live in `TestCases/OtherTests/` with different column sets
# and are copied into `TestCases/` in the canonical form `load_target` expects:
#
#   TPU   `TPU_highrise_14_middle.dat`  -> y, u, Iu          (2 QoIs)
#   WoW   `themisABL.dat`               -> y, u, Iu, Iv, Iw  (4 QoIs)
#
# which matches what paper section 7.2 says about the two cases.
#
# **Both files are at wind-tunnel model scale, not full scale.**  The paper
# describes a 200 m full-scale high-rise for TPU and a 20 m low-rise for WoW,
# but the .dat files carry the tunnel's own heights, so `H_build` here is the
# building height *in the units of the target file*, y_T/1.5, and the
# full-scale height is context rather than an input.  Getting this wrong would
# put the geometric scale factor out by the tunnel's own scale ratio.
#
#     python dev/13_wind_tunnel_optima.py

# %%
import os
import shutil
import sys

import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import load_target                                   # noqa: E402
import optcompare as oc                                          # noqa: E402

ALPHA_GRID = np.round(np.arange(0.25, 0.7301, 0.001), 5)
SRC = 'TestCases/OtherTests'
DST = 'TestCases'

# name -> (source file, QoI columns, selection set)
TUNNELS = {
    'TPU': ('TPU_highrise_14_middle', ['u', 'Iu'], ['u', 'Iu']),
    'WoW': ('themisABL', ['u', 'Iu', 'Iv', 'Iw'], ['u', 'Iu', 'Iw']),
}


# %% [markdown]
# ## 1. Canonicalise the two targets

# %%
for name, (src, qois, _) in TUNNELS.items():
    d = pd.read_csv(f'{SRC}/{src}.dat')
    keep = ['y'] + qois
    missing = [c for c in keep if c not in d.columns]
    assert not missing, f'{src}: missing {missing}'
    out = d[keep].sort_values('y').reset_index(drop=True)
    out.to_csv(f'{DST}/{name}.dat', index=False)
    print(f'{name:<4} {len(out)} points, y in '
          f'[{out.y.min():.4f}, {out.y.max():.4f}] m, QoIs {qois}')


# %% [markdown]
# ## 2. Optimise

# %%
cache = oc.load_cache('dev/cache_pod.npz')
rows = []
for name, (_, qois, select) in TUNNELS.items():
    t, _, U_ref, y_T, y_ref = load_target(name)
    obj = oc.CachedObjective(cache, t, qois)
    g = oc.run_grid(obj, cache['xList'], ALPHA_GRID)
    g['score'] = sum(g[q] / g[q].min() for q in select)
    best = g.loc[g['score'].idxmin()]
    norms = {q: np.linalg.norm(obj.T[q]) for q in qois}
    H_build = y_T / 1.5
    rows.append({
        'bldg': name, 'cat': 'WT',
        'h': best['$h$'], 'r': int(best['$r$']), 'x_B': best['$x$'],
        's_U': best[r'$k$'], 'y_Max': best[r'$\alpha$'],
        'y_ref': y_ref, 'U_ref': U_ref, 'V': best[r'$k$'] * U_ref,
        'H_build': H_build, 'y_T': y_T,
        'n_qoi': len(qois), 'select': '+'.join(select),
        **{f'res_{q}': 100 * best[q] / norms[q] for q in qois},
    })
    print(f'  {name}: h={best["$h$"]:.2f} r={int(best["$r$"])} '
          f'x_B={best["$x$"]:.1f} s_U={best[r"$k$"]:.3f} '
          f'y_Max={best[r"$\alpha$"]:.3f}  H_build={H_build:.3f} m '
          f'(1:{y_T/best[r"$\alpha$"]:.1f})')

wt = pd.DataFrame(rows)
wt.to_csv('dev/wt_optima_pod.csv', index=False)
pd.set_option('display.width', 220, 'display.max_columns', 40)
print('\n--- wind-tunnel optima (residuals as % of the target norm) ---')
print(wt.round(3).to_string(index=False))

print('\n--- r boundary check (HANDOVER 19.3) ---')
for name, (_, qois, select) in TUNNELS.items():
    t, _, _, _, _ = load_target(name)
    obj = oc.CachedObjective(cache, t, qois)
    g = oc.run_grid(obj, cache['xList'], ALPHA_GRID)
    g['score'] = sum(g[q] / g[q].min() for q in select)
    gb = g.groupby('$r$')['score'].min()
    r_opt = int(wt.loc[wt.bldg == name, 'r'].iloc[0])
    print(f'  {name}: r_opt={r_opt}  on bound={r_opt == 52}  '
          f'argmin over r={int(gb.idxmin())}  '
          f'slope at 52={gb.loc[53.0] - gb.loc[52.0]:+.4f}  '
          f'span={100*(gb.max()-gb.min())/gb.min():.1f}%')


# %% [markdown]
# ## 3. Build the two cases
#
# TPU's velocity column is normalised (its values run 0.67 to 1.11 and pass
# through 1 near y_ref), so the case is written but flagged: `Vinlet` is
# dimensionless and the run is blocked until a reference wind speed is
# supplied.  WoW's column is in m/s, so that case is complete.

# %%
import caselib as cl                                             # noqa: E402

OUT = 'dev/cases'
rows = []
for _, w in wt.iterrows():
    name = w['bldg']
    dim = name != 'TPU'
    notes = () if dim else (
        'TPU velocities are normalised in the source data; supply the tunnel '
        'reference speed at y_ref before running.  HANDOVER 23.',)
    d = cl.write_case(OUT, name, name, w['h'], int(w['r']), w['x_B'],
                      w['y_Max'], w['s_U'], w['H_build'],
                      geo=1.0, dimensional=dim, notes=notes)
    rows.append({'case': name, 'h': w['h'], 'r': int(w['r']),
                 'x_B': w['x_B'], 'y_Max': w['y_Max'], 's_U': w['s_U'],
                 'scale_factor': 1.0, **d})

wtman = pd.DataFrame(rows)
wtman.to_csv(f'{OUT}/manifest_wt.csv', index=False)
print(f'\nwrote {len(wtman)} wind-tunnel cases to {OUT}/')
print(wtman[['case', 'h', 'r', 'x_B', 'y_Max', 's_U', 'building_scale',
             'Vinlet', 'n_inlet', 'n_alf',
             'velocity_is_dimensional']].round(3).to_string(index=False))
