# %% [markdown]
# # Which configuration produced the published Pareto fronts?
#
# Re-scoring `*_decision_variables.csv` with the committed code and the
# `intensities` models does **not** reproduce `*_pareto_front.csv` (max abs
# discrepancy ~0.44, i.e. an order of magnitude, not numerical noise).  Two
# independent signs that the published runs used a different configuration:
#
#   * the published alpha values run over [0.250, 0.727], while
#     `optimizeParameters.py` bounds alpha to [0.30, 0.70];
#   * none of the three candidate `xList` definitions in the repo brings the
#     re-scored objectives close.
#
# Prime suspect: a different model set.  Oak carries four --
# `intensities`, `intensities_12points`, `intensities_high`,
# `intensities_high_noA` -- and the README only says the first is "best".
#
# This script sweeps (testID x xList) and reports which combination, if any,
# reproduces the published numbers.  Until one does, no claim comparing a grid
# search against the published front is admissible.
#
# Needs the probe subset in ../GPRModels/:
#   {0p3,0p6}_{intensities_12points,intensities_high,intensities_high_noA}_{u,Iu}.pkl

# %%
import itertools
import os
import sys

import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import load_target                                   # noqa
import gridsearch as gs                                          # noqa
from hyperparametersGPR import xList as HP_XLIST                 # noqa

FNAME = 'MRB_Cat_B'
MODELDIR = '../GPRModels'

# Only score published solutions whose $x$ gene rounds to index 0.  Every
# candidate xList then needs exactly one station (0.6 for A, 0.3 for B and C),
# which keeps the probe download to a handful of pickles.  Note that
# intensities_12points has 72 files = 18 stations and is missing 0p3, so it was
# fitted on the 18-station list -- itself a clue about which run used what.
X_INDEX = 0
MAX_ROWS = 6

TESTIDS = ['intensities', 'intensities_12points',
           'intensities_high', 'intensities_high_noA']

XLISTS = {
    'A active (18, from 0.6)':
        [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0,
         3.3, 3.6, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0],
    'B commented (18, no 1.8)':
        [0.3, 0.6, 0.9, 1.2, 1.5, 2.1, 2.4, 2.7, 3.0,
         3.3, 3.6, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0],
    'C hyperparametersGPR (19)': list(HP_XLIST),
}

# %% [markdown]
# ## 1. What is actually on disk

# %%
have = sorted(f for f in os.listdir(MODELDIR) if f.endswith('.pkl'))
inv = {}
for f in have:
    body = f[:-4]
    for t in sorted(TESTIDS, key=len, reverse=True):
        if f'_{t}_' in body:
            x, q = body.split(f'_{t}_')
            inv.setdefault(t, {}).setdefault(x, []).append(q)
            break
for t in TESTIDS:
    d = inv.get(t, {})
    print(f'{t:24s} {len(d):>2d} stations  '
          f'{sorted(d)[:6]}{" ..." if len(d) > 6 else ""}')

# %% [markdown]
# ## 2. The published rows we are trying to reproduce

# %%
target, header, U_ref, y_T, y_ref = load_target(FNAME)
pub_all = pd.read_csv(f'TestCases/{FNAME}_pareto_front.csv')
dec_all = pd.read_csv(f'TestCases/{FNAME}_decision_variables.csv')

sel = np.flatnonzero(
    np.round(dec_all['$x$'].to_numpy()).astype(int) == X_INDEX)[:MAX_ROWS]
if len(sel) == 0:
    raise SystemExit(f'no published solution has $x$ index {X_INDEX}')
dec = dec_all.iloc[sel].reset_index(drop=True)
pub = pub_all.iloc[sel].reset_index(drop=True)
print(f'{len(dec)} published solutions with $x$ index {X_INDEX} '
      f'(of {len(dec_all)})\n')

print(dec.to_string())
print()
print('x gene -> index:', [int(np.round(v)) for v in dec['$x$']])
for name, xl in XLISTS.items():
    print(f'  {name:28s} -> ',
          [xl[int(np.round(v))] for v in dec['$x$']])
print()
print(pub.to_string())

# %% [markdown]
# ## 3. Sweep (testID x xList)
#
# Only `u` and `Iu` are scored -- those are the two QoIs the probe subset
# covers, and they are the pair the paper actually optimises.

# %%
QOIS = ['u', 'Iu']
rows = []
for tid, (xname, xl) in itertools.product(TESTIDS, XLISTS.items()):
    needed = {f'{str(xl[int(np.round(v))]).replace(".", "p")}_{tid}_{q}.pkl'
              for v in dec['$x$'] for q in QOIS}
    missing = [f for f in sorted(needed)
               if not os.path.exists(os.path.join(MODELDIR, f))]
    if missing:
        rows.append({'testID': tid, 'xList': xname, 'max abs': np.nan,
                     'mean abs': np.nan, 'note': f'missing {len(missing)} pkl'})
        continue
    try:
        chk = gs.reevaluate(target, QOIS, dec, xList=xl, testID=tid)
        d = np.abs(chk[QOIS].to_numpy() - pub[QOIS].to_numpy())
        rows.append({'testID': tid, 'xList': xname, 'max abs': d.max(),
                     'mean abs': d.mean(), 'note': ''})
    except Exception as exc:                                   # noqa: BLE001
        rows.append({'testID': tid, 'xList': xname, 'max abs': np.nan,
                     'mean abs': np.nan, 'note': repr(exc)[:60]})

res = pd.DataFrame(rows).sort_values('max abs')
print(res.to_string(index=False))

# %% [markdown]
# ## 4. Verdict

# %%
ok = res[res['max abs'] < 1e-6]
if len(ok):
    print('reproduced by:')
    print(ok.to_string(index=False))
else:
    best = res.dropna(subset=['max abs'])
    print('No combination reproduces the published front.')
    if len(best):
        print(f'closest: {best.iloc[0]["testID"]} / {best.iloc[0]["xList"]}  '
              f'max abs {best.iloc[0]["max abs"]:.4f}')
    print()
    print('Remaining suspects, in rough order of likelihood:')
    print('  1. TestCases/*.dat changed after the optimisation was run')
    print('     (git has the .dat and the CSVs in the same commit, so the repo')
    print('      cannot settle the order -- ask whoever ran it)')
    print('  2. a yMax other than 1.0')
    print('  3. a model set that never made it to Oak')
    print('  4. an older scale_predictions / objective definition')
