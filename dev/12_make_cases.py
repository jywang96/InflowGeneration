# %% [markdown]
# # Build every downstream case for the verification campaign
#
# Writes, for each selected setup, the two boundary-condition tables CharLES
# reads and the `caseConfig.json` that records the setup, into
# `dev/cases/<name>/`.  Meshing and the CharLES input deck stay on the server,
# because `modelDefinition.generateCase` shells out to `surfer.exe`.
#
# **The reason this script exists rather than `generateInflow.py`.**
# `generateInflow.py` loads `../GPRModels/{-4p95,-2p85}_inflow_stresses_*.pkl`,
# i.e. the superseded point-wise models, *and one set per plane*.  That
# contradicts paper section 4.3.1, which describes a single upstream surrogate
# pooled by true row index `n`, and it is the surrogate Figure 4 validates.
# Boundary conditions generated the old way would verify a framework the paper
# does not describe.  See HANDOVER 22.
#
# This script uses `podlib.PODSurrogate(features=('h','n'), log_cols=())`,
# queried at `n = r - 7` for the inlet and `n = r` for the ALF target, exactly
# as paper equation `eqn:upstreamqueries` states, and reports the difference
# from the old path so the ~8 per cent shift of HANDOVER section 2 is on record.
#
#     python dev/12_make_cases.py

# %%
import json
import os
import sys

import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from hyperparametersGPR import trainPairs                        # noqa: E402

OUT = 'dev/cases'


CASES = pd.read_csv('dev/asce_optima_pod.csv')
# Group A of HANDOVER 21: the MRB Cat D baseline and its three scaled repeats.
# The scale factors are the controlled variable; the resulting building scale
# 1:s_y follows from the selected y^Max and is not rounded.
SCALED = {'MRB_Cat_D': [1.0, 1.5, 3.0, 6.0]}
# Group B and C are single runs at their own optimum; everything in the CSV is
# built, and HANDOVER 21 says which subset to actually queue.
WANTED = ['MRB_Cat_D', 'LRB_Cat_B', 'MRB_Cat_B', 'MRB_Cat_C']


# %% [markdown]
# ## 1-2. Surrogate and case writer
#
# Both live in `dev/caselib.py` so this script and `13_wind_tunnel_optima.py`
# cannot drift apart.

# %%
import caselib as cl                                             # noqa: E402

cl.upstream_surrogate()
QU = cl.QU
Y_FIT = cl.Y_FIT
ROW_OFFSET = cl.ROW_OFFSET
upstream_bc = cl.upstream_bc


# %% [markdown]
# ## 3. Build them

# %%
H = {'LRB': 6.0, 'MRB': 30.0, 'HRB': 100.0}
rows = []
for _, w in CASES.iterrows():
    # HANDOVER 8: attribute access on a column called 'cat' hits the
    # pandas .cat accessor, so index it explicitly.
    fname = f"{w['bldg']}_Cat_{w['cat']}"
    if fname not in WANTED:
        continue
    for sc in SCALED.get(fname, [1.0]):
        name = fname if sc == 1.0 else f'{fname}_scale{sc:g}'
        d = cl.write_case(OUT, name, fname, w['h'], int(w['r']), w['x_B'],
                          w['y_Max'], w['s_U'], H[w['bldg']], geo=sc)
        rows.append({'case': name, 'h': w['h'], 'r': int(w['r']), 'x_B': w['x_B'],
                     'y_Max': w['y_Max'], 's_U': w['s_U'], 'scale_factor': sc,
                     **d})

man = pd.DataFrame(rows)
man.to_csv(f'{OUT}/manifest.csv', index=False)
pd.set_option('display.width', 220, 'display.max_columns', 40)
print(f'\nwrote {len(man)} cases to {OUT}/')
print(man[['case', 'h', 'r', 'x_B', 'y_Max', 's_U', 'scale_factor',
           'building_scale', 'Vinlet', 'n_inlet', 'n_alf']]
      .round(3).to_string(index=False))


# %% [markdown]
# ## 4. How far the new boundary conditions are from the old ones
#
# HANDOVER section 2 recorded that pooling the upstream planes shifts the ALF
# stresses by about 8 per cent, but the figure had never been quantified
# against the models that actually wrote the shipped boundary conditions.

# %%
print('\n--- new (pooled S^U) vs old (point-wise, per plane) ---')
try:
    import gridsearch as gs                                       # noqa: E402
    from modelDefinition import gaussianProcess                   # noqa: E402

    for plane, tag in ((-4.95, 'inlet'), (-2.85, 'ALF')):
        pfx = str(f'{plane}_').replace('.', 'p')
        gp = gaussianProcess(
            {'h': trainPairs[:, 0], 'r': trainPairs[:, 1], 'x': [plane]},
            {'h': trainPairs[:, 0], 'r': trainPairs[:, 1], 'x': [plane]},
            {'h': trainPairs[:, 0], 'r': trainPairs[:, 1], 'x': [plane]},
            1.0, './GPRDatabase', Y_FIT)
        for _, w in man[man.scale_factor == 1.0].iterrows():
            n = w['n_inlet'] if plane == -4.95 else w['n_alf']
            new = upstream_bc(w.h, n)
            ff = pd.DataFrame({'y': Y_FIT, 'x': plane,
                               'h': w['h'], 'r': w['r']})
            msg = []
            for q in QU:
                old = gp.predict(f'../GPRModels/{pfx}inflow_stresses_{q}.pkl',
                                 ff, ['y', 'h', 'r'], q)['y_model'].to_numpy()
                den = np.linalg.norm(old)
                if den > 0:
                    msg.append(f'{q} {100*np.linalg.norm(new[q]-old)/den:5.1f}%')
            print(f'  {tag:<5} {w["case"]:<22} ' + '  '.join(msg))
except Exception as e:
    print(f'  skipped ({type(e).__name__}: {e})')
    print('  the old models need ../GPRModels and ./GPRDatabase; '
          'this comparison is diagnostic only and does not affect the cases.')


# %% [markdown]
# ## 5. Pre-flight check: does each case actually hit its target at y_ref?
#
# The anchoring of paper eq (eqn:suref) makes the delivered profile agree with
# the target at y_ref *by construction*, so a discrepancy here means the setup
# and the target have drifted apart -- a stale optimum, a wrong y_ref, or a
# clipped s_U.  `generateInflow.py:plotABL` warns above 0.5 per cent; the same
# threshold is applied here, before anything is queued on the server.
#
# HANDOVER section 3 records what caused this last time: [0.8, 1.8] bounds on
# s_U left over from the old normalisation, silently clipping.

# %%
import optcompare as oc                                          # noqa: E402
from common import load_target                                   # noqa: E402

cache = oc.load_cache('dev/cache_pod.npz')
QD = ['u', 'Iu', 'Iv', 'Iw']
print('\n--- anchoring check (tolerance 0.5%) ---')
bad = 0
for _, w in man[man.scale_factor == 1.0].iterrows():
    t, _, U_ref, y_T, y_ref = load_target(w['case'])
    obj = oc.CachedObjective(cache, t, QD)
    i = obj.rows(w['h'], w['r'])
    M = obj.M(w['y_Max'])
    pU = M @ cache['Q'][w['x_B']]['u'][i]
    # s_U is anchored, so s_U * p_U(eta_ref) must equal t_U(eta_ref) = 1
    err = 100 * abs(w['s_U'] * pU[obj.j] - obj.T['u'][obj.j]) \
        / abs(obj.T['u'][obj.j])
    # and the dimensional velocity at y_ref must reproduce the target there
    V_at_ref = w['s_U'] * pU[obj.j] * U_ref
    flag = '' if err <= 0.5 else '   *** EXCEEDS TOLERANCE ***'
    bad += err > 0.5
    print(f"  {w['case']:<12} eta_ref={obj.eta_ref:.4f}  "
          f"anchor error {err:.3f}%  U(y_ref)={V_at_ref:.3f} m/s "
          f"(target {U_ref:.3f}){flag}")
print(f'\n{bad} case(s) outside tolerance')
if bad:
    raise SystemExit('anchoring check failed; do not queue these cases')
