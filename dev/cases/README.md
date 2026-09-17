# Downstream LES campaign — prepared 2026-09-09

Nine ready cases plus one that needs a number from you. Every setup comes from
the POD+GPR surrogate the paper describes; nothing here was produced by the
superseded point-wise models. Full rationale in `HANDOVER.md` sections 21–23.

## Read first

`HANDOVER.md` **section 15** is the operating brief — what the framework is,
why these runs matter, what to hand back. **Section 21** is the case list and
what each case buys. Section 22 explains why the boundary conditions were not
made with `generateInflow.py`. Section 8 lists the environment traps.

## What is in each directory

| file | what it is |
|---|---|
| `<case>_inflow_input.txt` | inlet mean velocity profile, `INLET_PROFILE` |
| `<case>_ALF_input.txt` | ALF Reynolds-stress targets, `DATA_ALF` |
| `caseConfig.json` | the setup, plus a `_derived` block of everything computed from it |

Both tables are tab-separated with the twelve columns `generateInflow.py`
writes, on the same 1501-point grid, so no deck change is needed.

`caseConfig.json` field names follow the existing convention, which is not
self-evident:

- **`alpha` is `y^Max`**, the truncation height in metres at model scale.
- **`k` is `s_U`**, the velocity scale factor.
- `y_ref` is a height in the units of the target's `y` column, not a ratio.
- `scaleFactors.scale` is what `modelDefinition` needs; `_derived.geometric_factor`
  is the physically meaningful one. See section 22.

## The cases

| directory | h [m] | r | x_B [m] | y^Max | s_U | building scale | group |
|---|---|---|---|---|---|---|---|
| `MRB_Cat_D` | 0.04 | 55 | 6.0 | 0.367 | 1.366 | 1:122.6 | A |
| `MRB_Cat_D_scale1.5` | 0.04 | 55 | 6.0 | 0.367 | 1.366 | 1:81.7 | A |
| `MRB_Cat_D_scale3` | 0.04 | 55 | 6.0 | 0.367 | 1.366 | 1:40.9 | A |
| `MRB_Cat_D_scale6` | 0.04 | 55 | 6.0 | 0.367 | 1.366 | 1:20.4 | A |
| `LRB_Cat_B` | **0.13** | 85 | 0.6 | 0.304 | 2.444 | 1:29.6 | B |
| `MRB_Cat_B` | **0.05** | 52 | 0.6 | 0.250 | 1.648 | 1:180.0 | C |
| `MRB_Cat_C` | **0.05** | 52 | 4.0 | 0.417 | 1.373 | 1:107.9 | C |
| `WoW` | 0.06 | 92 | 0.6 | 0.325 | 1.857 | 1:1.2 | D |
| `TPU` | 0.04 | 52 | 1.8 | 0.575 | 1.205 | 1:1.2 | D — **blocked** |

Bold `h` values are **not** simulated roughness heights; the database has
`h ∈ {0.04, 0.06, 0.08, 0.12, 0.14, 0.16}`. Those three cases test the
surrogate off the LES grid, which is the point of groups B and C.

The two wind-tunnel cases have `building scale ≈ 1:1.2` because their targets
are already at tunnel model scale, unlike the ASCE targets which are full
scale. That is expected, not a bug.

### `TPU` is blocked

Its target's velocity column is normalised in the source data (values run 0.67
to 1.11, passing through 1 near `y_ref`), so `_derived.Vinlet` is
dimensionless and the written profiles carry no physical scale. To unblock:

1. find the TPU reference wind speed at `y_ref = 0.466` m, model scale;
2. multiply every velocity column by it and every Reynolds-stress column by
   its square, in both `.txt` files;
3. record the number used in `caseConfig.json`.

The optimum itself is unaffected — the inverse problem is dimensionless — so
`h`, `r`, `x_B`, `y^Max`, `s_U` all stand.

## Run order

Priority is group A, then B, then C, then D. Group A alone delivers two paper
sections (7.5 and 7.6), so run `MRB_Cat_D` first and check it before queueing
the rest.

## Per case

1. Copy the directory somewhere writable and `cd` into it.
2. Mesh: `surfer.slurm` then `stitch.slurm`. `modelDefinition.generateCase`
   shells out to `surfer.exe`, which is why meshing was not done here.
3. Run: `charles.slurm`. It runs `charles_helm.exe` twice, a starter then the
   main run.
4. Match the database exactly — you are adding to it, not designing anew:
   `Δt = 0.0004` s, `Co ≈ 1`, 15.1 M control volumes, 30 s with statistics
   over the last 20 s, domain 22.25 × 3.0 m, ALF box inlet → 2.1 m, roughness
   2.25 → 4.95 m, 10 rows. All dimensions scale by
   `_derived.geometric_factor`.

**One caveat on the run length.** The deck `modelDefinition.py` emits has
`NSTEPS = 100000` at `Δt = 0.0004` (40 s) with `RESET_STATS TIME = 10` (a 30 s
window), which disagrees with the paper's 30 s / last 20 s. HANDOVER item M-10.
Use the paper's numbers and flag any discrepancy you find with what the
original 28 runs actually used — this is an open question for Mattia.

## Hand back

Per HANDOVER 15.6. In addition, for every group A case give the **three-curve
decomposition** explicitly — target, surrogate prediction, LES — because that
panel is the entire point of paper section 7.5:

- LES vs surrogate prediction = surrogate error
- surrogate prediction vs target = inverse-problem residual

Predicted residuals to compare against are in
`../asce_optima_pod.csv` and `../wt_optima_pod.csv`, as percentages of the
target norm.

Also state explicitly whether the ~8 % ALF stress shift of HANDOVER 22.2 is
visible in the result. That is the second question these runs settle.

## Group E is not here

Extending the database below `r = 52` needs raw upstream CharLES output under
`../../PFCoarseABL/PFh<h>/`, which is not on the laptop. Run
`python dev/14_extend_r_server.py --probe` on the server; it prints the two
`formatSetup.py` entries to add, verbatim, and checks them. See HANDOVER 21 E
and 23.

## Do not

- Do not re-run the 28 database cases.
- Do not "improve" a setup. Run exactly what the inverse step selected; if a
  selection looks wrong, report it separately.
- Do not regenerate boundary conditions with `generateInflow.py`. It reads the
  superseded models. Use `dev/12_make_cases.py` and `dev/13_wind_tunnel_optima.py`.
- Do not use `sed` on `.tex`.
