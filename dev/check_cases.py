#!/usr/bin/env python
"""Sanity-check the generated boundary-condition tables before queueing them.

Run on the laptop before packing, and again on the server after unpacking, so
a transfer problem cannot go unnoticed.

    python dev/check_cases.py [cases_dir]

What is checked, and what is deliberately not:

  * **A non-monotone mean velocity is expected**, not an error.  ABL profiles
    over a roughness canopy carry a wake inflection near the ground, which
    paper section 4.1 names as one of the features the learned basis has to
    reproduce.  A first attempt at this script flagged every file for
    non-monotonicity and was wrong to.
  * What *would* be wrong is a large reversal or a high-amplitude ripple, so
    the peak-to-trough excursion is measured against U_max instead.  The POD
    truncation leaves a ripple of a few tenths of a per cent, far below the
    +/-10 per cent band the paper assesses against; the gradient changes sign
    many times inside that ripple, which is why counting sign changes is not
    a useful test on its own.
  * Reynolds-stress signs: uu, vv, ww must be non-negative and uv negative,
    which is the convention generateInflow.py writes and CharLES expects.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CASES = sys.argv[1] if len(sys.argv) > 1 else 'dev/cases'
RIPPLE_TOL = 3.0        # per cent of U_max; POD ripple is ~0.9
FAIL = []


def check(path):
    d = pd.read_csv(path, sep='\t')
    u = d['x-velocity'].to_numpy()
    out = []

    if not np.isfinite(d.to_numpy()).all():
        out.append('non-finite values')
    if (u < 0).any():
        out.append('negative velocity')
    if u.max() <= 0:
        out.append('velocity identically zero')

    # peak-to-trough excursion below the running maximum
    ripple = 100 * (np.maximum.accumulate(u) - u).max() / max(u.max(), 1e-30)
    if ripple > RIPPLE_TOL:
        out.append(f'ripple {ripple:.2f}% of U_max exceeds {RIPPLE_TOL}%')

    for c in ('uu', 'vv', 'ww'):
        if (d[f'{c}-reynolds-stress'] < 0).any():
            out.append(f'{c} negative')
    if (d['uv-reynolds-stress'] > 0).any():
        out.append('uv positive (should be negative)')

    y = d['y'].to_numpy()
    if not np.all(np.diff(y) > 0):
        out.append('y not strictly increasing')

    return ripple, out


print(f'{CASES}: checking\n')
print(f'{"case":<20}{"file":<30}{"y range":<20}{"U range":<18}'
      f'{"ripple":>8}  status')
for path in sorted(glob.glob(os.path.join(CASES, '*', '*_input.txt'))):
    case = os.path.basename(os.path.dirname(path))
    d = pd.read_csv(path, sep='\t')
    u = d['x-velocity'].to_numpy()
    ripple, issues = check(path)
    if issues:
        FAIL.append((path, issues))
    yr = f'[{d.y.min():.3f}, {d.y.max():.2f}]'
    ur = f'[{u.min():.2f}, {u.max():.2f}]'
    print(f'{case:<20}{os.path.basename(path):<30}{yr:<20}{ur:<18}'
          f'{ripple:7.2f}%  ' + ('  '.join(issues) if issues else 'ok'))

# every case must carry a config, and a blocked one must say so
print()
for cfg in sorted(glob.glob(os.path.join(CASES, '*', 'caseConfig.json'))):
    c = json.load(open(cfg))
    case = os.path.basename(os.path.dirname(cfg))
    der = c.get('_derived', {})
    if not der.get('velocity_is_dimensional', True):
        print(f'{case:<20}BLOCKED: {der.get("BLOCKED", "")[:80]}')
    for k in ('h', 'r', 'alpha', 'k', 'x', 'y_ref'):
        if k not in c.get('reference', {}):
            FAIL.append((cfg, [f'reference.{k} missing']))

print(f'\n{len(FAIL)} problem(s)')
for p, iss in FAIL:
    print(f'  {p}: {"; ".join(iss)}')
sys.exit(1 if FAIL else 0)
