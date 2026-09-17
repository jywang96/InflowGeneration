# %% [markdown]
# # P0-2: rebuild the optimisation cache from the POD+GPR surrogate
#
# `cache_high_noA.npz` was built from the superseded point-wise pickles in
# `../GPRModels`, so Figure 9 and Table 2 of the paper were computed with the
# surrogate the paper says it replaced, while Figures 4-6 validate the new one.
# See HANDOVER 11.1.  This script rebuilds the cache from `PODSurrogate` and
# reports what moved.
#
# Four outputs:
#   1. `dev/cache_pod.npz`, the new cache
#   2. the nine ASCE optima, and a side-by-side against the published table
#   3. **whether MRB Cat D moved** -- this decides whether the four
#      geometric-scaling LES of paper section 7.6 are stale (HANDOVER 11.1)
#   4. the reachable range of I_vert / I_stream over the *full enumerated
#      grid*, not just the 380 training profiles (HANDOVER 11.7)
#
#     python dev/11_rebuild_cache_pod.py

# %%
import os
import sys
import time

import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

from common import load_target                                   # noqa: E402
import gridsearch as gs                                          # noqa: E402
import optcompare as oc                                          # noqa: E402

CACHE = 'dev/cache_pod.npz'
QD = ['u', 'Iu', 'Iv', 'Iw']
SELECT = ['u', 'Iu', 'Iw']            # I_vert is unreachable; HANDOVER 11.7
ALPHA_GRID = np.round(np.arange(0.25, 0.7301, 0.001), 5)
CATS = ['B', 'C', 'D']
BLDG = ['LRB', 'MRB', 'HRB']

# Published Table 2, from the old point-wise cache.  Kept only to report the
# delta; nothing downstream should read these numbers.
OLD = {
    ('LRB', 'B'): (0.07, 74, 0.6, 1.62, 0.34),
    ('LRB', 'C'): (0.04, 62, 1.5, 1.53, 0.26),
    ('LRB', 'D'): (0.04, 79, 3.6, 1.54, 0.28),
    ('MRB', 'B'): (0.04, 67, 0.6, 1.60, 0.25),
    ('MRB', 'C'): (0.04, 45, 3.6, 1.32, 0.31),
    ('MRB', 'D'): (0.04, 53, 7.0, 1.36, 0.30),
    ('HRB', 'B'): (0.04, 50, 1.8, 1.30, 0.38),
    ('HRB', 'C'): (0.04, 48, 9.0, 1.33, 0.25),
    ('HRB', 'D'): (0.04, 51, 13.0, 1.30, 0.35),
}
SIMULATED_H = [0.04, 0.06, 0.08, 0.12, 0.14, 0.16]


# %% [markdown]
# ## 1. Build the cache

# %%
t0 = time.time()
cache = oc.build_cache_pod(CACHE, n_modes=5)
print(f'\ncache ready in {time.time() - t0:.1f}s')
print(f'  {len(cache["xList"])} stations x {len(cache["HR"])} (h,r) '
      f'x {cache["Q"][cache["xList"][0]]["u"].shape[1]} heights')


# %% [markdown]
# ## 2. Re-run the nine ASCE cases

# %%
rows = []
for b in BLDG:
    for c in CATS:
        t0 = time.time()
        t, _, U_ref, y_T, y_ref = load_target(f'{b}_Cat_{c}')
        obj = oc.CachedObjective(cache, t, QD)
        g = oc.run_grid(obj, cache['xList'], ALPHA_GRID)

        # Selection rule, HANDOVER 11.5: min-normalised sum over SELECT.
        # This is the rule the paper never states; P0-6 writes it down.
        g['score'] = sum(g[q] / g[q].min() for q in SELECT)
        best = g.loc[g['score'].idxmin()]

        norms = {q: np.linalg.norm(obj.T[q]) for q in QD}
        rows.append({
            'bldg': b, 'cat': c,
            'h': best['$h$'], 'r': int(best['$r$']), 'x_B': best['$x$'],
            's_U': best[r'$k$'], 'y_Max': best[r'$\alpha$'],
            'y_ref': y_ref, 'U_ref': U_ref, 'V': best[r'$k$'] * U_ref,
            **{f'res_{q}': 100 * best[q] / norms[q] for q in QD},
        })
        print(f'  {b} Cat {c}: h={best["$h$"]:.2f} r={int(best["$r$"])} '
              f'x_B={best["$x$"]:.1f} s_U={best[r"$k$"]:.3f} '
              f'y_Max={best[r"$\alpha$"]:.3f}   ({time.time() - t0:.1f}s)')

new = pd.DataFrame(rows)
new.to_csv('dev/asce_optima_pod.csv', index=False)
print('\nwrote dev/asce_optima_pod.csv')

pd.set_option('display.width', 200, 'display.max_columns', 40)
print('\n--- new optima (residuals as % of the target norm) ---')
print(new.round(3).to_string(index=False))


# %% [markdown]
# ## 3. What moved, and is the scaling study stale?

# %%
print('\n--- old (point-wise GPR) vs new (POD+GPR) ---')
print(f'{"case":<10} {"h":>14} {"r":>10} {"x_B":>12} {"s_U":>14} {"y_Max":>14}')
moved = []
for _, w in new.iterrows():
    key = (w['bldg'], w['cat'])
    oh, orr, ox, ok, oa = OLD[key]
    same = (abs(oh - w['h']) < 5e-3 and orr == w['r']
            and abs(ox - w['x_B']) < 1e-6)
    if not same:
        moved.append(key)
    print(f'{w["bldg"]+" "+w["cat"]:<10} '
          f'{oh:.2f}->{w["h"]:.2f}'.rjust(15)
          + f'{orr}->{w["r"]}'.rjust(11)
          + f'{ox:.1f}->{w["x_B"]:.1f}'.rjust(13)
          + f'{ok:.2f}->{w["s_U"]:.2f}'.rjust(15)
          + f'{oa:.2f}->{w["y_Max"]:.2f}'.rjust(15)
          + ('' if same else '   MOVED'))

print(f'\n{len(moved)} of 9 setups moved: {moved}')

mrb_d = new[(new.bldg == 'MRB') & (new.cat == 'D')].iloc[0]
oh, orr, ox, ok, oa = OLD[('MRB', 'D')]
stale = not (abs(oh - mrb_d['h']) < 5e-3 and orr == mrb_d['r']
             and abs(ox - mrb_d['x_B']) < 1e-6)
print('\n--- HANDOVER 11.1 / 15.4: the geometric-scaling study ---')
print(f'  paper section 7.6 ran 4 LES at MRB Cat D = '
      f'h={oh}, r={orr}, x_B={ox}, y_Max={oa}')
print(f'  the POD optimum is            h={mrb_d["h"]:.2f}, '
      f'r={mrb_d["r"]}, x_B={mrb_d["x_B"]:.1f}, y_Max={mrb_d["y_Max"]:.3f}')
print('  => ' + ('STALE: those 4 LES are no longer at the optimum.  Either '
                 're-run them\n     or reframe 7.6 as a representative setup '
                 'drawn from the database.'
                 if stale else
                 'UNCHANGED: the 4 scaling LES remain valid as they stand.'))

print('\n--- HANDOVER 15.4: which cases have an unsimulated h? ---')
for _, w in new.iterrows():
    d = min(abs(w['h'] - hh) for hh in SIMULATED_H)
    flag = '  <-- interpolated, strongest verification candidate' if d > 5e-3 else ''
    print(f'  {w["bldg"]} Cat {w["cat"]}: h={w["h"]:.2f}, '
          f'nearest simulated {d:.2f} away{flag}')


# %% [markdown]
# ## 4. Reachable range of I_vert / I_stream over the full enumerated grid
#
# ASCE 49-21 prescribes 0.500.  The claim in HANDOVER 11.7 was computed over
# the 380 training profiles; recomputing it over every enumerable candidate
# makes it a statement about the framework's reachable set rather than about
# the training sample.

# %%
ratios = []
for x in cache['xList']:
    Iu, Iv = cache['Q'][x]['Iu'], cache['Q'][x]['Iv']
    ok = Iu > 1e-6
    ratios.append(np.where(ok, Iv / np.where(ok, Iu, 1.0), np.nan))
R = np.concatenate(ratios, axis=0)              # (n_candidates, n_y)

prof = np.nanmean(R, axis=1)                    # one scalar per profile
print(f'candidate profiles: {prof.size}  '
      f'({len(cache["xList"])} stations x {len(cache["HR"])} (h,r))')
print('\nprofile-mean I_vert / I_stream:')
for p in [0, 1, 5, 25, 50, 75, 95, 100]:
    print(f'  {p:>3}th percentile  {np.nanpercentile(prof, p):.3f}')
print(f'\npointwise over all candidates and heights: '
      f'[{np.nanmin(R):.3f}, {np.nanmax(R):.3f}]')
print(f'ASCE 49-21 requirement: 0.500')
print(f'candidates with profile-mean ratio <= 0.500: '
      f'{int(np.sum(prof <= 0.5))} of {prof.size}')

np.save('dev/ivert_ratio_pod.npy', prof)
print('\nwrote dev/ivert_ratio_pod.npy')
print(f'\ntotal {time.time() - t0:.1f}s')
