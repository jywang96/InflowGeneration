# %% [markdown]
# # Group E of HANDOVER 21: extend the upstream database below r = 52
#
# **This script must run on the server.**  It needs the raw upstream CharLES
# output under `../../PFCoarseABL/PFh<h>/`, which is not on the laptop; only
# the 28 already-formatted `GPRDatabase/*.txt` files are.  Everything else in
# the campaign was prepared locally; this is the one step that cannot be.
#
# ## Why
#
# Seven of the nine ASCE optima, and the TPU optimum, land exactly on
# `r = 52`, the most upstream station the downstream database contains, and the
# objective increases monotonically as `r` leaves it (HANDOVER 19.3).  The
# unconstrained optimum is therefore outside the database, and the paper
# currently cannot say by how much.  One downstream LES at an `r < 52` recipe
# answers that.
#
# ## What is free and what is not
#
# Extracting new upstream stations is **free**: the six upstream LES already
# resolve all 92 rows, so a station at `n = 38` is a plane that was simulated
# and merely never post-processed.  What costs a run is the *downstream* LES,
# because `r` enters only through the boundary conditions.
#
# ## The station bookkeeping
#
# `r` labels the ALF plane; the inlet plane is `n = r - 7`.  For the target
# recipe `r = 45`:
#
#     inlet plane   n = 38   ->  pseudo-coordinate -4.95
#     ALF plane     n = 45   ->  pseudo-coordinate -2.85
#
# `n = 45` already exists in the database as the inlet plane of `r = 52`, so
# only `n = 38` is genuinely new.  That is the check to run first: if the two
# extractions of `n = 45` do not agree, the station lookup is wrong.
#
# ## Before touching formatSetup.py
#
# Read the pseudo-coordinate comment in `formatSetup.py` around line 60.  It
# records three independent proofs of the current mapping, which was once
# swapped, mislabelling the inlet profile as the ALF target.  The `else: raise`
# at the end of that block is deliberate.  Do not remove it -- extend the
# lookup tables instead.
#
#     python dev/14_extend_r_server.py --probe     # report what is available
#     python dev/14_extend_r_server.py --extract   # write the new profiles

# %%
import os
import sys

import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

MODE = ('--extract' if '--extract' in sys.argv else '--probe')

# The prefixes in formatSetup encode the plane's streamwise position in metres
# with 'p' for the decimal point.  Rows are 0.3 m apart, so
#
#     x(n) = 23.65 + 0.3 * (n - 52)
#
# anchored on rDictALF['52'] = '23p65'.  This was verified rather than assumed:
# it reproduces all nine rDictALF prefixes and all nine rDictInflow prefixes
# under n = r - 7, and in particular it sends n = 45 to '21p55', which is
# exactly rDictInflow['52'].  The inlet plane of label 52 *is* station 45, and
# that coincidence is the check that the formula and the r -> n convention
# agree.
#
# The two entries to add for TARGET_R = 45 are therefore, verbatim:
#
#     rDictInflow['45'] = '19p45'      # n = 38, the only genuinely new plane
#     rDictALF['45']    = '21p55'      # n = 45, already in the database
#
# so only one new plane per height has to exist in the raw output.
TARGET_R = 45                     # the recipe to add
ROW_OFFSET = 7
NEW_STATIONS = [TARGET_R - ROW_OFFSET, TARGET_R]      # 38 and 45
H_LIST = [0.04, 0.06, 0.08, 0.12, 0.14, 0.16]
RAW = '../../PFCoarseABL/PFh{h:.2f}/'


# %% [markdown]
# ## Probe: what stations exist in the raw output?

# %%
def read_lookups():
    """Parse rDictInflow and rDictALF out of formatSetup.py.

    formatSetup.py has no `if __name__ == '__main__'` guard: importing it runs
    the whole conversion and fails as soon as the raw output is missing.  The
    tables are read statically instead, which also works on the laptop.
    """
    import ast
    src = open('formatSetup.py', encoding='utf8').read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ('rDictInflow', 'rDictALF'):
                out[t.id] = ast.literal_eval(node.value)
    missing = {'rDictInflow', 'rDictALF'} - set(out)
    assert not missing, f'could not parse {missing} from formatSetup.py'
    return out['rDictInflow'], out['rDictALF']


def probe():
    """List the collapse files present, so the prefix for each n can be read off."""
    rDictInflow, rDictALF = read_lookups()

    print('formatSetup lookup tables, for reference:')
    print('  rDictInflow (inlet plane, pseudo-coordinate -4.95):')
    for k in sorted(rDictInflow, key=int):
        print(f'    r={k:<3} -> prefix {rDictInflow[k]}')
    print('  rDictALF (ALF plane, pseudo-coordinate -2.85):')
    for k in sorted(rDictALF, key=int):
        print(f'    r={k:<3} -> prefix {rDictALF[k]}')

    print('\nThe prefixes encode the streamwise position in metres with "p"')
    print('for the decimal point, e.g. 21p55 = 21.55 m.  Two consecutive')
    print('labels differ by 5 rows = 1.5 m; inlet and ALF of the same label')
    print('differ by 7 rows = 2.1 m.  Both relations are verifiable above,')
    print('and together they give the prefix for any n:')
    known_r = min(int(k) for k in rDictALF)
    known_x = float(rDictALF[str(known_r)].replace('p', '.'))
    print(f'    x(n) = {known_x} + 0.3 * (n - {known_r})')
    for n in NEW_STATIONS:
        x = known_x + 0.3 * (n - known_r)
        print(f'    n={n:<3} -> {x:.2f} m -> prefix '
              f'{("%.2f" % x).replace(".", "p")}')

    print('\nraw output present?')
    for h in H_LIST:
        d = RAW.format(h=h)
        if not os.path.isdir(d):
            print(f'  h={h:.2f}  MISSING  {d}')
            continue
        files = sorted(f for f in os.listdir(d) if 'collapse_width' in f)
        pfx = sorted({f.split('_avg_')[0] for f in files})
        print(f'  h={h:.2f}  {len(files):>4} collapse files, '
              f'{len(pfx)} prefixes')
        for n in NEW_STATIONS:
            x = known_x + 0.3 * (n - known_r)
            want = ('%.2f' % x).replace('.', 'p')
            print(f'      n={n:<3} prefix {want:<7} '
                  f'{"PRESENT" if want in pfx else "ABSENT"}')


# %% [markdown]
# ## Extract: add the new stations to the database
#
# The safe edit to `formatSetup.py` is to extend the two lookup tables with the
# new label, not to bypass them:
#
#     rDictInflow['45'] = '<prefix for n = 38>'
#     rDictALF['45']    = '<prefix for n = 45>'
#
# then run `formatSetup.py` for `r = 45` across the six heights.  It writes
# `GPRDatabase/PFh<h>u15r45.txt` in the same format as the existing 28, and
# `podlib.build_upstream_snapshots` picks the new stations up with no change,
# because it indexes by `n` and not by label.
#
# The consistency check that must pass: `n = 45` is already in the database as
# the inlet plane of label 52, so extracting it again under label 45 must
# reproduce it.

# %%
def check_duplicate_station():
    """n = 45 exists twice once r = 45 is added; the two must agree."""
    import podlib as pl
    QU = ['u', 'uu', 'vv', 'ww', 'uv']
    Y = np.linspace(0.01, 1.0, 400)
    meta, X = pl.build_upstream_snapshots(
        np.array([[h, 52] for h in H_LIST] + [[h, 45] for h in H_LIST]),
        QU, Y, 1.5)
    bad = 0
    for h in H_LIST:
        m = (meta['h'] == h) & (meta['n'] == 45)
        if m.sum() != 2:
            print(f'  h={h:.2f}: {m.sum()} copies of n=45, expected 2')
            bad += 1
            continue
        i, j = np.flatnonzero(m.to_numpy())
        for q in QU:
            d = np.linalg.norm(X[q][i] - X[q][j]) / max(
                np.linalg.norm(X[q][i]), 1e-30)
            if d > 1e-6:
                print(f'  h={h:.2f} {q}: two extractions of n=45 differ by '
                      f'{100*d:.3f}%  <-- station lookup is wrong')
                bad += 1
    print(f'  {"OK" if not bad else str(bad) + " MISMATCHES"}')
    return bad == 0


if MODE == '--probe':
    probe()
    print('\nnext, add these two entries to formatSetup.py verbatim:')
    print("    rDictInflow['45'] = '19p45'   # n = 38, the new plane")
    print("    rDictALF['45']    = '21p55'   # n = 45, = rDictInflow['52']")
    print('then run formatSetup.py for r=45 and re-run this with --extract')
else:
    if not os.path.exists('GPRDatabase/PFh0.04u15r45.txt'):
        raise SystemExit(
            'GPRDatabase/PFh0.04u15r45.txt is absent: run formatSetup.py for '
            'r=45 first (see the docstring above)')
    print('checking the duplicate station n=45')
    if not check_duplicate_station():
        raise SystemExit('station lookup check failed; do not proceed')
    print('\nthe database now contains r=45.  To build the case:')
    print('  1. add 45 to gridsearch.R_GRID (it is arange(52, 93))')
    print('  2. add (h, 45) rows to hyperparametersGPR.trainPairs for the')
    print('     heights extracted, so S^U is fitted with the new stations')
    print('  3. rm dev/cache_pod.npz && python dev/11_rebuild_cache_pod.py')
    print('  4. python dev/12_make_cases.py')
    print('  5. compare the HRB Cat C residual against the r>=52 value in')
    print('     dev/asce_optima_pod.csv; that difference is the result')
