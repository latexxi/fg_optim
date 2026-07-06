# RESULT — Task 04, the M-sweep (`mesh/validate_adapt.py`)

Run: `python3 -m mesh.validate_adapt`. All numbers below are from that run, this
commit, `Ms=(16,32,64,128)`, `k_seed=4`, `n_band=4` (the spec's stated defaults).
Nothing here is hand-picked after the fact — this is the script's actual stdout.

## 1. The four diagnosis findings this rests on (03-driver.md §3.0)

1. **Cold-starting a deep grid lands in a random stuck basin.** `alternating_maximization`
   is coordinate ascent on a bilinear objective with many fixed points; a cold solve at
   high `(N,M)` is not a reliable optimum.
2. **Only a disciplined climb from `k=1` is trustworthy.** Uniform doubling threads
   the good basin; nothing here ever cold-starts a deep grid (mandatory Phase A).
3. **At fixed `M`, x-refinement saturates.** Seeded from a healthy basin, both uniform
   and band-refine jump once, then flatten. A fixed-M `dJk -> 0` is an M-ceiling
   artifact, not evidence of bounded `J` — hence this task.
4. **Band-refinement is a real but partial efficiency lever.** It reaches most of the
   x-gain at a fraction of the nodes, not all of it — the honest number is below, not
   assumed.

## 2. Gates (04-validate.md §4.2) — all four PASS

```
[GATE 1] Phase-A basin fidelity (M=32, k=1..5): max|Jc_adapt - J_dyadic| = 0.000e+00 -> PASS
[GATE 2] Monotonicity + feasibility (M=32, k_seed=4, n_band=4): worst dJk = -1.865e-14,
         all-gen feasible = True -> PASS
[GATE 3] Band-mass premise (|x|<0.4): min frac over 8 generations = 1.0000 -> PASS
         per-gen fracs: 1.000 x8 (every generation, 100% of harvest mass inside the band —
         no leakage into the arms at any depth reached here)
[GATE 4] Band-vs-uniform at matched N (band N=107 vs uniform N=129, M=32):
         Jc_band=2.473941  Jc_uniform=2.583677  ratio=0.9575 -> PASS (informative, threshold 0.80)
```

Gate 1/2 passing means everything below is reportable per §4.2's own rule (a verdict
from a gate-1/2 failure would not be). Gate 3 is stronger here than diagnosis expected
(100%, not just >=95%) — at the depths this sweep reaches (N<=107 in-band), the arms
genuinely carry zero curvature; leakage may still appear at a depth this sweep doesn't
reach. Gate 4's ratio (0.9575) is *better* than diagnosis's earlier ~87% figure, but it's
one matched-N point at M=32, `n_band=4` — not a general constant (§3, node-efficiency
table below tells a messier story once M varies).

## 3. `Jc_sat(M)` — the central experiment (§4.1)

```
  M | Jc_sat(M) | N_at_sat | nodes_at_sat | elapsed
 16 |  2.450602  |    107   |    1819      |  0.3s
 32 |  2.473941  |    107   |    3531      |  1.3s
 64 |  2.485692  |    107   |    6955      |  2.6s
128 |  2.753053  |    107   |   13803      | 14.7s
```

Sequence: `[2.4506, 2.4739, 2.4857, 2.7531]` — **monotone non-decreasing in `M`**, so
gate 1/2's discipline holds and the sequence clears `m_sweep`'s own basin-trouble guard
(no flag raised). No RAM wall was hit at these settings (`k_seed=4, n_band=4` tops out at
`N=107`; the deepest matrix, `M=128`, solved in 14.7s) — the practical M=128 ceiling
this task's spec worried about did not bind here; a much deeper `n_band` would be needed
to find it.

**Read: monotone, but the M=128 jump (`+0.267`, dwarfing the `+0.023`/`+0.012` steps
before it) does not by itself support "rising, unbounded" over "leveling, bounded" —
it needs a caveat that the per-M ladders make directly visible and that changes the
honest reading from "clean rising trend" to inconclusive-but-suggestive:**

Looking at the raw per-M generation tables (script output, Phase A rows only):

```
 M= 16  Phase A:  2.000, 2.000, 2.2875, 2.2875      (escapes the J=2.0 plateau at k=3, N=9)
 M= 32  Phase A:  2.000, 2.000, 2.1600, 2.1732      (escapes at k=3, N=9)
 M= 64  Phase A:  2.000, 2.000, 2.0000, 2.0000      (never escapes within k<=4 — still 2.0 at N=17!)
 M=128  Phase A:  2.000, 2.000, 2.0000, 2.5603      (escapes at k=4, N=17 — one step later than 16/32, but a much bigger jump)
```

`M=64`'s disciplined climb is *still stuck at the trivial `J=2.0` fixed point through
all of Phase A* (`k=1..4`) at this `k_seed`; it only escapes once Phase B's band
refinement perturbs the grid (`+0.4588` at Phase B gen 4). `M=128`'s climb escapes one
generation later than `M=16/32` but jumps much further when it does (`+0.560` at
`k=4` alone). So the four `Jc_sat(M)` points are not "the same basin, sampled at four
resolutions" — each `M` finds (or fails to find) the escape from the coordinate-ascent's
shallow fixed point at a *different* generation, and the size of that escape jump swamps
the smooth per-M trend the money plot is trying to show. The monotone-in-M sequence
survives `m_sweep`'s guard (it IS monotone), but the guard only checks
non-decreasing-ness, not that the points are basin-comparable — and by inspection here
they are not fully comparable at this `k_seed`.

**Honest verdict: INCONCLUSIVE, leaning-rising, not "bounded."** The sequence rises
monotonically and gate 1/2 hold, so it is not disqualified outright — but the escape-
timing artifact above means the `M=128` point's size cannot be trusted as a clean
"deeper `M` genuinely buys more `J`" reading; it may equally be "deeper `M` let this
particular disciplined climb escape a shallow fixed point it would otherwise have
stayed stuck in, at this `k_seed`." Distinguishing those needs a deeper `k_seed`
(so every `M`'s Phase A has escaped the `J=2.0` plateau before Phase B or the M-compare
even starts) rerun at this same `Ms` tuple — not done here (that is the direct next
step, not a new hypothesis). Per §4.5's explicit instruction, this is **not** rounded
down to "bounded" — the leveling-looking gap between the `M=16/32/64` points and the
`M=128` point is exactly the kind of noisy-but-monotone signature the spec says not to
over-read.

## 4. Band-vs-uniform node efficiency (§4.3)

Target `Jc* = 2.473941` (the `M=32` band ceiling from this run):

```
   M | uniform: (N, nodes)              | band: (N, nodes)                 | node ratio
  16 | not reached by k=7 (best 2.4488) | not reached (best 2.4506)        | n/a
  32 | (  65,   2145)                   | ( 107,   3531)                   | 0.61
  64 | not reached by k=7 (best 2.4089) | ( 107,   6955)                   | n/a
 128 | (  17,   2193)                   | (  17,   2193)                   | 1.00
```

Read this table carefully — it is **not** a clean "band wins" table, and stating that
would misrepresent it:

- **`M=32` (the row this ratio is defined against): ratio 0.61 — uniform reaches the
  target at *fewer* nodes than band**, the opposite of the efficiency claim's naive
  reading. This is not a bug: the target `Jc*` was itself extracted from the `M=32`
  band route, and gate 4 (§2 above, a *different*, matched-`N` comparison) already
  shows uniform beats band at equal N by only ~4% (ratio 0.9575) — so of course uniform
  needs fewer *total* nodes to reach a value band only reaches at N=107.
- **`M=128`: ratio 1.00 is degenerate, not a real tie.** `M=128`'s Phase A alone
  overshoots the target at `N=17` (`Jc=2.5603 > 2.4739`, see §3's escape-jump finding)
  — both "uniform" and "band" routes report the identical Phase-A point because the
  target was already cleared before Phase B (band) ever ran. This is the same basin-
  escape-timing artifact from §3, surfacing here as a table artifact — not a genuine
  band-vs-uniform comparison.
- **`M=16` and `M=64`: neither route reaches the target within the tested depth.**
  `M=16` cannot reach `2.4739` by either route (consistent with §3 — M is a real
  constraint, not just a knob to spend nodes on). `M=64`'s *uniform* route is capped at
  2.4089 (worse than `M=32`'s uniform ceiling, again the escape-timing artifact —
  `M=64`'s disciplined climb is stuck at the shallow fixed point through `k=7`), while
  its *band* route does reach the target — showing band recovering value a stuck
  uniform climb at that same `M` missed, not "band is more efficient" in the resolution
  sense §4.3 asks about.

**Honest reading:** the one clean, non-degenerate efficiency read is gate 4's own
matched-`N` number: **band captures ~96% of uniform's `Jc` at matched `N`, for ~83%
of the nodes (`N=107` vs `N=129` at `M=32`)** — a real but partial win, in line with
diagnosis's ~87% figure (this run's number is somewhat better, one data point, not a
re-derivation of a constant). The §4.3 table's cross-M "nodes to reach a fixed target"
framing is contaminated by the same per-M basin-escape-timing issue as §3 and should not
be read as a clean efficiency ladder without a deeper `k_seed` first.

## 5. The caveat (per §4.5, mandatory)

Everything above measures the **same hierarchy** the uniform mesh (`refine_baseline.py`)
does, just reached more cheaply and deeper: band-refine only *relocates* x-nodes into
the region gate 3 confirms already carries ~100% of the harvest mass, and the disciplined
Phase-A climb only *selects* which coordinate-ascent basin the rest of the run inherits.
Neither step is new physics — no constraint, objective, or LP changed. The M-sweep is
therefore a real but narrow probe: it answers "does the *ceiling this specific
construction reaches* move with `M`", not "is `sup J` over all feasible `(f,g)` bounded."
Given the basin-escape-timing confound found in §3-4, even that narrower question is not
yet cleanly answered by this run — the honest state is **inconclusive, leaning toward
"M matters and the ceiling isn't flat," pending a deeper-`k_seed` rerun to remove the
escape-timing confound before trusting the `M=128` point's magnitude.**

## 6. Post-hoc resolution: the fix is basin discipline in M, not a deeper k_seed

Follow-up probes (main-thread, after the task run) settle §3's open question, and the
answer is sharper than "rerun at deeper `k_seed`":

**6a. The M-axis is basin-fragile even for a fully disciplined UNIFORM climb.**
`dyadic_refinement(k_start=1, k_max=7)` at each M gives `Jc(k=7)`:

```
 M= 16  k1..k7:  2.000 2.000 2.288 2.287 2.287 2.425 2.449
 M= 32  k1..k7:  2.000 2.000 2.160 2.173 2.284 2.584 2.584
 M= 64  k1..k7:  2.000 2.000 2.000 2.000 2.248 2.375 2.409
 M=128  k1..k7:  2.000 2.000 2.000 2.560 2.642 2.670 2.720
```

`Jc(k=7)` vs M = `2.449, 2.584, 2.409, 2.720` — **non-monotone (M=64 < M=32)**. So the
confound is NOT cured by climbing deeper in k: each M seeds the *same* g-ramp init and
escapes the trivial `J=2.0` fixed point at a different k (M=64 not until k=5), landing
wherever coordinate ascent happens to catch. Fixed-M and cold M-sweeps are basin noise,
full stop. The §3 `Jc_sat(M)` "rise" is not trustworthy.

**6b. Warm-starting ACROSS M (basin discipline in the time axis too) rescues it.**
Climb k=1→6 at a small M=16 to a healthy basin, then grow M by warm-starting each step
(`regauge_time` the previous solution onto the finer time grid, re-alternate), fixed k=6:

```
 M= 16  Jc=2.4248
 M= 32  Jc=2.5097
 M= 64  Jc=2.5280
 M=128  Jc=2.6064
 M=256  Jc=2.7522
```

**Clean monotone rise**, and the late increments *grow* (`+0.078` at 64→128, `+0.146`
at 128→256) rather than decaying — no leveling through M=256. Same k, same everything
except the basin is carried across M instead of re-cold-started, and the M=64 dip is
gone. This is the basin-clean version of §3's experiment.

**Corrected verdict.** The `mesh/` coordinate-ascent LP alternation only yields a
trustworthy resolution ladder under **2-D basin discipline** — warm-start across *both*
k (x-refinement) and M (time-refinement). Under that discipline the J ladder rises
monotonically with M with no sign of a ceiling through M=256, which **leans unbounded**
(consistent with the mesh's original `J ~ ln(resolution)` motivation). This is one k=6
slice, so it is suggestive, not proof — the x-resolution still caps it. The clean next
experiment is now well-defined and un-confounded: a **full 2-D basin-disciplined climb**
(alternately warm-start-refine k and M from a tiny `(k,M)` seed, band-refine x for the
efficiency win), reading J against joint resolution. `refine_adapt.py` already does the
k half and `prolong.regauge_time` already does the M-warm-start half; wiring the M-climb
into the driver is the remaining step. Until that runs, the honest headline is:
**fixed-M/cold sweeps are basin artifacts; the basin-disciplined ladder leans unbounded.**

## 7. The full 2-D basin-disciplined climb — Phase C wired in, run

The remaining step from §6 is done: `adaptive_refinement(..., n_mclimb=)` adds **Phase C**
— from the converged Phase-B seed (basin-disciplined in k, band-refined x), freeze x and
climb M by warm-starting across it (`adaptive_warm_start` -> `regauge_time` stretches the
field over index space, `prolong_x` identity since x frozen), re-alternate each doubling.
This is the un-confounded J-vs-resolution read the whole track was built for: both axes
basin-disciplined, x-nodes spent efficiently in the band.

Run `adaptive_refinement(k_seed=4, n_band=3, k0=1, M=16, n_mclimb=4)`:

```
 gen | ph |  N  |  M  | nodes |   Jc     |   dJk
   2 |  A |   9 |  16 |   153 | 2.28750 | +0.2875   (Phase A basin climb)
   4 |  B |  23 |  16 |   391 | 2.37161 | +0.0841   (Phase B band depth ...
   5 |  B |  35 |  16 |   595 | 2.41851 | +0.0469    ... saturates at M=16:)
   6 |  B |  59 |  16 |  1003 | 2.41851 | +0.0000
   7 |  C |  59 |  32 |  1947 | 2.52531 | +0.1068   (Phase C M-climb breaks
   8 |  C |  59 |  64 |  3835 | 2.64105 | +0.1157    through the M-16 ceiling,
   9 |  C |  59 | 128 |  7611 | 2.73100 | +0.0900    x frozen at N=59 band grid)
  10 |  C |  59 | 256 | 15163 | 2.86434 | +0.1333
```

**Phase B saturates** (dJk -> 0 at fixed M=16, exactly finding 3), then **Phase C's
M-climb reignites the ladder**: `Jc = 2.419 -> 2.525 -> 2.641 -> 2.731 -> 2.864`, clean
monotone, and the increments **do not decay** — the last one (`+0.133`, M 128->256) is the
*largest* of the four. Same qualitative signature as §6b's uniform-x M-climb, but now on a
band-refined x grid (N=59, arms frozen) reaching M=256 at 15k nodes instead of a uniform
grid's ~33k, and with x depth already spent where the harvest is. Every generation is
monotone (asserted in the driver) and every warm start feasible (asserted in
`adaptive_warm_start`).

**Final verdict (both axes basin-disciplined, un-confounded).** Under 2-D basin discipline
the J ladder rises monotonically with resolution with **no ceiling and non-decaying
increments through M=256 / N=59** — this **leans unbounded**, consistent with
`J ~ ln(resolution)`. It is still one band-x slice climbed in M; a genuinely-2-D grid
(interleave more k-refinement between M-doublings) would tighten it further, but the
confounds that made §3's fixed-M/cold sweeps unreadable (basin scatter, M-position
artifacts) are now all removed, and the honest signal is a clean rising ladder that does
not level off.

## 8. Genuinely-2-D interleaved climb (`two_d_climb`) — the proof-grade read

§7's Phase C freezes x and climbs only M (one band-x slice). `two_d_climb` removes that
last confound: after a Phase-A/B seed, each step grows **both** x-band-resolution AND M
in a *single* warm start (`adaptive_warm_start` = `regauge_time` over the finer time-index
grid + `prolong_x` inserting the new band nodes, both J-neutral, feasibility asserted),
then re-alternates. So the two axes climb in lockstep — the joint `(x, M)` resolution
limit, both basin-disciplined.

Run `two_d_climb(k_seed=3, n_band_seed=1, k0=1, M0=16, n_steps=5)`:

```
 gen | ph |  N  |  M  | nodes  |   Jc     |   dJk   | solve
   2 |  A |   9 |  16 |    153 | 2.28750 | +0.2875 |   0.0s
   3 |  B |  13 |  16 |    221 | 2.35893 | +0.0714 |   0.0s
   4 |  D |  19 |  32 |    627 | 2.39205 | +0.0331 |   0.0s  (first joint step, transient)
   5 |  D |  31 |  64 |   2015 | 2.58196 | +0.1899 |   0.3s  (both N and M doubling ...
   6 |  D |  55 | 128 |   7095 | 2.74266 | +0.1607 |   1.8s   ... per-octave increment
   7 |  D | 103 | 256 |  26471 | 2.89537 | +0.1527 |  63.3s   settles near-constant ...
   8 |  D | 199 | 512 | 102087 | 3.06775 | +0.1724 | 886.7s   ... and holds one deeper)
```

Past the first transient step, the per-octave increment holds at **~0.15–0.19,
near-constant and not decaying** across a genuine joint-resolution doubling (N: 31→55→103→199,
M: 64→128→256→512) — the last increment (`+0.1724`, gen8) is *not* smaller than gen6/gen7's.
That flat per-octave gain is exactly the `J ~ ln(resolution)` log-growth signature — the
same shape the reference uniform mesh showed (`+0.21`/octave, k04→06), now reproduced on a
basin-disciplined, band-efficient joint grid reaching **Jc=3.068 at N=199/M=512/102k nodes**
(crossing the reference uniform mesh's k06=3.055 at far fewer nodes). A *bounded* J would
require this increment to decay toward zero as resolution grows; through five octaves it
does not.

**Cost / the wall is TIME, not RAM (measured).** The per-octave solve time (both axes ×2,
nodes ×4) grows super-linearly as the LP densifies: `0.3 → 1.8 → 63.3 → 883.6s` (gen5→8).
gen8 alone is ~15 min at 102k nodes. The gen9 octave (M=1024 / N=391 / ~400k nodes) was
launched detached with a 7 h budget: it did **not** OOM (RAM held), but a single
`alternating_maximization` solve there did **not converge in ~6.75 h** — the dense HiGHS LP
per alternation × up to 80 alternations is the bottleneck (per-iter cost >~300 s at 400k
nodes). So gen8 (5 octaves, Jc=3.068) is the practical depth for this dense-LP solver; a
6th octave needs either a capped-iteration lower-bound solve (a not-fully-converged
alternation is still feasible and a valid J *lower* bound — enough to test "does the ladder
keep rising"), a warm-restarted/sparser LP path, or a multi-day run. The `check_feasible`
tol at the deep warm start was also loosened `1e-9 → 1e-6` (the arms sit exactly at
|slope|=1, and float64 interp roundoff there — measured 8.7e-8 at N=391 — false-positived
the feasibility assert; the LP re-solve projects exact regardless).

**Track verdict.** Across all three basin-disciplined reads (§6b uniform-x M-climb, §7
Phase C band-x M-climb, §8 genuine 2-D interleave) the J ladder rises monotonically with
resolution with **near-constant, non-decaying per-octave increments and no ceiling** —
consistently **leaning unbounded** (`J ~ ln(res)`). This is not a proof (finite resolution
can never rule out a very-late ceiling, and coordinate ascent only certifies a lower bound
per grid, not the global optimum), but every confound that made the earlier fixed-M/cold
sweeps unreadable has been removed and the surviving signal points one way. The honest
headline: **basin-disciplined climbing on both axes gives a clean log-growth ladder with
no sign of saturation through N=199 / M=512 (Jc=3.068, five octaves).**
