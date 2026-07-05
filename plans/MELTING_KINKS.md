# Finding: "Melting" kinks — the structure of the J>3 grid optimizer

## Status: EMPIRICAL FINDING from mesh data (`~/fg_opt3/data/level_k0*.npz`),
not yet reproduced constructively. This documents what the high-J grid
solutions actually look like and why they beat the kink prototype, so the
constructive arm can be rebuilt to match (see "Next step" at the end).

## For a standalone reasoner (self-contained pointer)

- **Data files:** `~/fg_opt3/data/level_k01.npz` ... `level_k08.npz`
  (absolute: `/home/lauri/fg_opt3/data/level_k0{1..8}.npz`).
- **Each file** (`numpy.load`, keys): `f`, `g` — arrays of shape
  `(Nx, Mt)` indexed `[x_index, t_index]`, giving f(x,t) and g(x,t) sampled
  on the grid; `x_grid` (length Nx, spans [-1,1]); `t_grid` (length Mt, spans
  [0,1]); `k` (level integer); `J` (scalar, the achieved objective).
- **The single decisive file:** `level_k06.npz`, `Nx=65`, `Mt=257`,
  **J = 3.05529** — the highest J in the set and the one all the structural
  plots/quantities below are read from. It is a certified-feasible point of

      max J[f,g] = int_0^1 int_{-1}^1 f_t(x,t) * g_xx(x,t) dx dt

  under: x-convex f,g; |f_x|,|g_x| <= 1; f_t >= 0; g_t <= 0;
  f(+-1,t)=g(+-1,t)=0; f(x,1)=0; g(x,0)=0. J>3 matters because tent
  (single-kink) functions provably cap at J=2 — so J=3.055 is a strict,
  numerically-certified breach of the tent cap, and the mechanism for that
  breach is the "melting" described below.
- Reconstruct any quantity in this doc from `level_k06.npz` alone via the
  snippet at the bottom ("Reproduce these numbers"); nothing here depends on
  this repo's code.

## Source data

Mesh optimizer runs (rectangular `Nx × Mt` grid, `f`/`g` sampled on it),
saved in `~/fg_opt3/data/level_k0k.npz` (keys `f, g, x_grid, t_grid, k, J`;
`f`/`g` shape `(Nx, Mt)`, indexed `[x, t]`). The problem is the same coupled
functional this repo attacks (see `~/Downloads/fg_opt-1.pdf`):

    max J[f,g] = int_0^1 int_{-1}^1 f_t(x,t) * g_xx(x,t) dx dt

with the identical constraint dictionary (x-convex, Lipschitz |.|_x <= 1,
f_t >= 0, g_t <= 0, f(+-1,t)=g(+-1,t)=0, f(x,1)=0, g(x,0)=0).

| level | Nx  (=2^k+1) | Mt  | J       |
|-------|--------------|-----|---------|
| k01   | 3            | 129 | 2.0000  |
| k02   | 5            | 129 | 2.0000  |
| k03   | 9            | 129 | 2.0000  |
| k04   | 17           | 129 | 2.6253  |
| k05   | 33           | 129 | 2.8361  |
| k06   | 65           | 257 | 3.0553  |
| k07   | 129          | 129 | 2.9788  |
| k08   | 257          | 65  | 2.8450  |

The mesh reaches **J = 3.055** (k06), far above the kink prototype's best
optimized run (2.52, Run 5) and the Run 11 constructive toy (0.31). So the
mesh finds structure the limited-K local search in this repo does not.

## Headline: the optimal slices are NOT tents — they "melt"

At every level the boundary slices are **exact full-depth centered hats**
(`min f(.,0) = min g(.,1) = -1.0000` to 4 dp, apex at x=0) — confirmed on
every k. This matches the analytic expectation: the per-x rise budget is
`int_0^1 f_t(x,t) dt = -f(x,0)`, maximized by making f(.,0) the deepest
Lipschitz-convex shape = the full hat; symmetric for g via g(.,1).

The INTERIOR slices are where the action is, and they are not tents. Compare
one f-slice at t=0 (tent) vs t=0.5 (melted), from k06:

| quantity                    | t=0 (tent) | t=0.5 (melted) |
|-----------------------------|-----------:|---------------:|
| depth (-min f)              | 1.000      | 0.683          |
| arm slope (Lipschitz)       | +-1.000    | +-0.957        |
| curvature-support width     | 0.00 (pt)  | **0.47** (band)|
| total slope-change (gxx mass) | 2.000    | 1.622          |
| largest single-cell share of curvature | **1.00** | **0.25** |

**Definition of melting.** The tent tip is a single Dirac curvature delta —
all the slope change (2, Lipschitz-saturated on both arms) packed at one
point. "Melting" = that delta **spreads into a distributed curvature band**
over an x-interval (width ~0.47 here; largest single cell holds only 25% of
the mass, i.e. spread over ~4 cells), and that band **drifts in x** as t
advances (peak location -0.03 -> ~-0.2 -> back toward 0 over the run). The
straight Lipschitz-1 arms are preserved (f stays a slope-1 wedge from each
boundary), only the sharp V-bottom becomes a wide, rounded, moving basin.

A melted slice is literally a **multi-kink convex function** (many hats side
by side smeared into a drifting arc), not one tent.

## Why melting sucks more into J

Crucial invariant: **total curvature mass per slice stays <= 2** at all t
(Lipschitz cap: 1.62 at t=0.5, 2.0 at the boundary). Melting adds NO
curvature mass. It **redistributes a fixed <=2 mass in space and time.**

`J = int int f_t * g_xx`. The rigid tent parks its whole g_xx delta at x=0,
so it only ever harvests f_t at x=0; the accounting telescopes to exactly 2
(Observation 1 in the PDF, Run 1 here). The melted g **sweeps** its curvature
band across x, so over the full time it overlaps high-f_t at every x it
visits — and each new x still has its untapped rise budget `-f(x,0)=1-|x|`
waiting. This is why the boundary slice must be the full hat: it stocks the
budget the melt then sweeps through. Same mass, far more overlap with f_t =>
bigger integral. This is the `L/w` amplification (STRATEGY.md Section 3)
restated geometrically: **melt = spread the curvature thin (small w) and drag
it far (large L).** Dually, f melts so its f_t is a tall, narrow, MOVING spike
co-located with g's drifting band, keeping the harvest hot everywhere the band
goes.

## Why it cascades (J>3, and maybe -> infinity)

The melt is **self-similar**: the coarse drifting basin carries its own finer
melt sub-band riding inside it (visible as branching sub-filaments in the
`f_t*g_xx` harvest heatmap), which carries a finer one still. Each dyadic
octave = one more generation of finer/faster melt.

Per-level J increment, where BOTH Nx and Mt are refined together:
- k04 -> k05 (Nx 17->33, both Mt=129):        **dJ = +0.211**
- k05 -> k06 (Nx 33->65 AND Mt 129->257):     **dJ = +0.219**

Two nearly-constant increments ~ **0.215 per dyadic level**. That is the log-
law signature: `J ~ 2 + 0.215*(level-3)` -> grows like `0.215*log2(Nx)`, and
`H[f] = int_0^1 sup_x f_t dt` grows with it (measured: 1.0 tents -> 2.53 ->
2.68 -> 3.26). The PDF's open question 2 ("does H[f] -> infinity?") is exactly
"can the melt keep sharpening?"

Restated in melt language:
> H[f] = int sup_x f_t dt. Melting makes f_t a tall-narrow moving spike, so
> sup_x f_t per t can grow as the melt sharpens (w -> 0). H[f] -> infinity
> IFF the melt can keep sharpening without the **re-arm cost** (maintaining
> the finer band consumes the coarser band's convex budget) eating the per-
> octave gain. The mesh's constant +0.215 says it does NOT get eaten — over
> two clean octaves.

## The budget-artifact confound (same trap as Run 9/10)

The "constant +0.215" rests on only TWO clean increments, because the
measurement is budget-sensitive in exactly the way Runs 9-10 documented:
**t-resolution must scale WITH x-resolution or J goes backward.**
- k07 (Nx=129, Mt=129) = 2.979 < k06 (Nx=65, Mt=257) = 3.055 — refining x
  while starving t goes DOWN.
- k08 (Nx=257, Mt=65) = 2.845 — badly t-starved.
The fine travelling melt-fronts need fine time to be resolved as they drift;
under-resolved t makes the finer generation look worthless. So the clean-
octave requirement (refine Nx and Mt together) is load-bearing, and the mesh
can only afford two such octaves before cost explodes. Two constant increments
is suggestive of the log law but NOT proof — the same "3 points can't
distinguish constant from decaying" situation as Run 9, one level up in
absolute J.

## Reconciliation with Run 11's "bounded" constructive result

Run 11 (`kink_opt/construct.py`) concluded bounded J. This mesh finding
contradicts it directionally, and the reason is the caveat Run 11 already
flagged: **the construction anchored every generation at ONE collapsing point
and used ONE kink per generation** — a rigid spike cascade, the exact opposite
of a broad drifting curvature band. It banned the melt (the sweep), so it saw
saturation. The mesh optimizer does the opposite — the harvest band drifts to
genuinely new x each pass, reaping each location's independent rise budget —
so it sees log-growth. Sweeping/melting is the load-bearing move; Run 11's toy
forbade it.

The melted slices ARE representable in this repo's hat basis (a melt = a sum
of many hats). The prototype missed J>3 for two concrete reasons:
1. **K too small.** 3-6 kinks cannot resolve a smooth drifting curvature band
   PLUS its finer sub-generations; the mesh has ~64 kinks/slice.
2. **Wrong construction.** Run 11 built a point-collapsing spike cascade, not
   a finite-width drifting melt.

## Next step (the construction that could turn the 2-point trend into a proof)

Rebuild the constructive arm so each generation is a **drifting curvature band
of finite width** (a melt), NOT a point:
- generation k = a curvature band of width `w_k ~ scale_x^k` that DRIFTS along
  an x-path of length `L_k` over a lifetime window `~ scale_t^k`, riding on
  generation k-1's band (not collapsing to k-1's endpoint);
- keep f(.,0), g(.,1) pinned to full hats (the budget stock);
- solve weights by the convex LP only (no position NLP — the Run 11 virtue);
- read the per-octave dJ analytically and check whether it stays constant as
  k -> infinity (constant => H[f] -> infinity, sup J = +infinity; decaying =>
  bounded, with the re-arm cost identified as the mechanism).

This is the one construction that is (a) artifact-free (LP-only, like Run 11)
AND (b) actually shaped like the mesh optimum (drifting melt, unlike Run 11) —
so it could PROVE rather than merely suggest either side, which neither the
mesh (cost-capped at 2 octaves) nor Run 11 (wrong shape) can.

## Reproduce these numbers

```
# boundary hats + H[f] growth + per-level dJ:  (reads ~/fg_opt3/data/)
python3 - <<'PY'
import numpy as np
for k in range(1,9):
    z=np.load(f'/home/lauri/fg_opt3/data/level_k0{k}.npz')
    f,g=z['f'],z['g']
    Hf=np.sum(np.max(np.diff(f,axis=1),axis=0))
    print(k, z['x_grid'].size, z['t_grid'].size, float(z['J']),
          Hf, f[:,0].min(), g[:,-1].min())
PY
```
