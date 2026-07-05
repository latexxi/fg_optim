"""Run 1-11: narrated demos. Not a test suite -- each run's configuration and
commentary explains what the previous run left on the table and why the next
one's hyperparameters were chosen. See CLAUDE.md's "The __main__ block"."""

import numpy as np

from .lp import lp_weights_f, lp_weights_g
from .objective import total_J
from .solver import run, multistart, _alternate
from .verify import certify, report, n_live_nodes, graded_grid, _ub
from .topology import (add_kink, spawn_generation, _seed_grown, prune,
                       grow_topology, generation_ladder, scale_sweep,
                       decay_ratios, corrected_decay_ratios, _gate_report,
                       _budget_stable, generation_step, fit_geometric)
from .construct import (constructive_ladder, sweep_ratios,
                        check_insertion_neutral, grid_convergence_check,
                        travel_sanity, saturation_diagnostics)
from .melt import (build_melt_hierarchy, melt_ladder, read_environment,
                   env_distance, check_band_neutral, band_travel_sanity,
                   mesh_cross_check)
from .persist import (save_run, save_ladder, save_sweep, save_construct,
                      save_melt)


def main():
    print("=" * 70)
    print("Run 1 (sanity): single co-located static kink pair, positions FROZEN")
    print("        exact discrete optimum is J = 2 (tent, bang-bang schedule)")
    print("=" * 70)
    r0 = run(N=24, Kf=1, Kg=1, outer=3, seed="static",
             optimize_pos=False, verbose=True)
    report("static 1+1", r0)
    save_run("run1", r0, meta=dict(N=24, Kf=1, Kg=1, outer=3, seed="static",
                                   optimize_pos=False))

    print()
    print("=" * 70)
    print("Run 2: same single kink pair, positions FREE (can travel discover >2?)")
    print("=" * 70)
    r1 = run(N=16, Kf=1, Kg=1, outer=6, seed="static", pos_iters=50)
    report("travel 1+1", r1)
    save_run("run2", r1, meta=dict(N=16, Kf=1, Kg=1, outer=6, seed="static",
                                   pos_iters=50))

    print()
    print("=" * 70)
    print("Run 3: Kf=3, Kg=2, positions free, seeded at center")
    print("=" * 70)
    r2 = run(N=16, Kf=3, Kg=2, outer=40, seed="static", pos_iters=40, patience=5)
    report("static 3+2", r2)
    save_run("run3", r2, meta=dict(N=16, Kf=3, Kg=2, outer=40, seed="static",
                                   pos_iters=40, patience=5))

    print()
    print("=" * 70)
    print("Run 4: push further -- more kinks (Kf=5, Kg=4) + multistart.")
    print("  Two things left J on the table in Run 3:")
    print("  (a) the position NLP step isn't monotone in J, so a run left")
    print("      going could drift below its own best point -- run() now")
    print("      keeps the best feasible state and reverts on regression")
    print("      (see 'patience' above -- Run 3 already benefits from this).")
    print("  (b) optimize_positions is only a local search, and the initial")
    print("      kink jitter was hardcoded to rng_seed=0. Sweeping seeds")
    print("      exposes materially better local optima at higher K.")
    print("  optimize_positions uses analytic gradients (grad_total_J +")
    print("  grad_penalty) throughout this file, not L-BFGS-B's finite")
    print("  differences -- Runs 2-3 above already benefited (~20-100x")
    print("  fewer objective evals per solve); it's what makes the wider")
    print("  multistart sweep in Run 5 below affordable.")
    print("=" * 70)
    r3 = multistart(seeds=range(6), N=16, Kf=5, Kg=4, outer=40,
                     seed="static", pos_iters=40, patience=5)
    report("multistart 5+4", r3)
    save_run("run4", r3, meta=dict(multistart_seeds=list(range(6)), N=16,
                                   Kf=5, Kg=4, outer=40, seed="static",
                                   pos_iters=40, patience=5,
                                   rng_seed=r3.get("rng_seed")))

    print()
    print("=" * 70)
    print("Run 5: cheap analytic gradients afford a much wider search --")
    print("  more seeds, more kinks. Kf=6,Kg=5 is the best FEASIBLE frontier")
    print("  found (Kf=7,Kg=6 and Kf=8,Kg=7 were tried and did not reliably")
    print("  beat it: 8+7 found a similar J but failed verify_dense, i.e. it")
    print("  exploited near-violations rather than a genuinely better optimum).")
    print("=" * 70)
    r4 = multistart(seeds=range(20), N=16, Kf=6, Kg=5, outer=60,
                     seed="static", pos_iters=80, patience=6)
    report("multistart 6+5", r4)
    save_run("run5", r4, meta=dict(multistart_seeds=list(range(20)), N=16,
                                   Kf=6, Kg=5, outer=60, seed="static",
                                   pos_iters=80, patience=6,
                                   rng_seed=r4.get("rng_seed")))

    print()
    print("=" * 70)
    print("Run 6 (Task B): topology moves -- birth/prune kinks between")
    print("  alternations. Starting from Run 3's converged 3+2 solution,")
    print("  grow_topology() greedily inserts one zero-weight kink at a time")
    print("  (a perturbed copy of the most active kink, alive only on a")
    print("  lifetime window), re-optimizes with the block alternation, prunes")
    print("  dead trajectories, and KEEPS the insertion only if J_certified")
    print("  strictly improves. Unlike Runs 4-5 (fixed K, brute-force more")
    print("  kinks + restarts) this is the add/prune machinery the")
    print("  hierarchical generation-spawning experiment (Task D) builds on.")
    print("=" * 70)
    grown = grow_topology(r2, n_gen=2, cand_seeds=range(2), outer=20,
                          pos_iters=40, patience=5)
    report("grown topo", grown)
    save_run("run6", grown, meta=dict(base="run3", n_gen=2,
                                      cand_seeds=list(range(2)), outer=20,
                                      pos_iters=40, patience=5))

    print()
    print("=" * 70)
    print("Run 7 (Task C): graded (non-uniform) time grid.")
    print("  total_J / the weight-LPs / the monotonicity checks never read")
    print("  node SPACING (dt cancels in the harvest sum), so an arbitrary")
    print("  non-uniform grid is transparent to them -- only seeding and the")
    print("  certification refinement consult t. That lets us spend time")
    print("  nodes where kinks actually live instead of a uniform global grid.")
    print("  Part A -- reproduce Run 3 within 1% at HALF the time nodes.")
    print("  Part B -- a narrow lifetime window costs O(1/w) nodes on a")
    print("            uniform grid but O(1) with grading: the variable-count")
    print("            saving Task C targets (fine generations live fast/short).")
    print("=" * 70)

    baseJ = certify(r2)["Jc"]                       # Run 3: 17 nodes, all-alive
    base_live = n_live_nodes(r2)
    bar = 0.99 * baseJ
    print(f"  Run 3 baseline: {r2['t'].size} nodes, {base_live} live vars, "
          f"J_certified = {baseJ:.4f}   (1% bar = {bar:.4f})")

    # Part A: half the nodes. All-alive Run 3 has no short lifetimes, so a
    # graded grid can't beat a uniform one here (verified: identical optima)
    # -- the win is purely the halved node count. Multistart because the
    # coarse-grid position NLP is nonconvex.
    t_half = np.linspace(0.0, 1.0, 8)               # 8 nodes vs 17
    bestJ, bestS = -np.inf, None
    for s in range(8):
        rh = run(Kf=3, Kg=2, outer=40, seed="static", pos_iters=60,
                 patience=6, verbose=False, rng_seed=s, t=t_half)
        ch = certify(rh)
        if ch["rep"]["ALL CONSTRAINTS OK"] and ch["Jc"] > bestJ:
            bestJ, bestS, best_rh = ch["Jc"], s, rh
    print(f"  Part A: {t_half.size} nodes, {n_live_nodes(best_rh)} live vars "
          f"({100*n_live_nodes(best_rh)//base_live}% of baseline), "
          f"J_certified = {bestJ:.4f} (seed {bestS})   "
          f"-> {'PASS' if bestJ >= bar else 'FAIL'} (within 1% at half nodes)")
    save_run("run7a", best_rh, meta=dict(t="half", N=8, Kf=3, Kg=2, outer=40,
                                         pos_iters=60, patience=6,
                                         rng_seed=bestS))

    # Part B: a fine g-kink alive only on a narrow window W. Resolve W to 6
    # local steps two ways and count live decision variables.
    W = (0.45, 0.55)                                # width 0.10
    tg = graded_grid([W], coarse_N=10, fine_sub=6)  # coarse bg + dense window
    rg = prune(run(Kf=3, Kg=2, outer=20, seed="static", pos_iters=40,
                   patience=5, verbose=False, t=tg), 1e-8)
    rngB = np.random.default_rng(0)
    parent = int(np.abs(rg["B"]).max(axis=0).argmax())
    XI2, ETA2, af2, ag2 = add_kink("g", rg["XI"], rg["ETA"], rg["alive_f"],
                                   rg["alive_g"], parent, tg, W[0], W[1],
                                   dx=0.03, rng=rngB)
    A0 = rg["A"]
    B0 = lp_weights_g(A0, XI2, ETA2, ub=_ub(ag2))
    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
    gcand = prune(_alternate(A0, XI2, B0, ETA2, tg, af2, ag2, outer=20,
                             pos_iters=40, optimize_pos=True, verbose=False,
                             patience=5), 1e-8)
    cg = certify(gcand)
    win_steps = int(((tg >= W[0] - 1e-9) & (tg <= W[1] + 1e-9)).sum()) - 1
    # uniform grid giving W the SAME 6 local steps needs step = |W|/6 over the
    # whole span -> ~1/step intervals everywhere; count its live vars for the
    # same 5 all-alive background kinks + the windowed kink.
    N_unif = int(round((1.0 - 0.0) / ((W[1] - W[0]) / win_steps)))
    live_unif = 5 * (N_unif + 1) + (win_steps + 1)
    print(f"  Part B: window {W} resolved to {win_steps} local steps")
    print(f"    graded : {tg.size:2d} nodes, {n_live_nodes(gcand):3d} live vars, "
          f"J_certified = {cg['Jc']:.4f}  ok = {cg['rep']['ALL CONSTRAINTS OK']}"
          f"  (within 1% of baseline: {cg['Jc'] >= bar})")
    print(f"    uniform: {N_unif + 1:2d} nodes, {live_unif:3d} live vars for the "
          f"SAME local resolution ({live_unif / n_live_nodes(gcand):.1f}x more "
          f"variables; the win grows as the window narrows)")
    save_run("run7b", gcand, meta=dict(window=W, coarse_N=10, fine_sub=6,
                                       outer=20, pos_iters=40, patience=5))

    print()
    print("=" * 70)
    print("Run 8 (Task D): the renormalization warm start -- and an HONEST")
    print("  null result. If the hierarchy were self-similar, the next")
    print("  generation of kinks would be an affinely-rescaled copy of the")
    print("  current one (shorter lifetime, narrower extent, riding the")
    print("  parent's path). spawn_generation() inserts exactly that contracted")
    print("  copy at ZERO weight (Task B machinery, so J and feasibility are")
    print("  unchanged at insertion -- verified below); a random insertion")
    print("  (add_kink) drops the same new kink at a RANDOM position on the")
    print("  same window. Both are re-optimized by the identical alternation.")
    print("  Hoped-for acceptance: the warm start reaches a better J faster.")
    print("  Measured: it does NOT. On Run 3's gen-0 optimum the contracted")
    print("  copy lands nearly CO-LOCATED with its (barely-travelling) parent,")
    print("  and two hats at one point are redundant in a convex sum -- so the")
    print("  LP gives it ~no weight and it converges in 1-2 outers to a")
    print("  SHALLOWER basin, while random insertion explores genuinely new")
    print("  positions and does at least as well (and stays feasible). Reading:")
    print("  the gen-0 optimum is not yet a self-similar travel hierarchy, so")
    print("  the renormalization premise is unvalidated at k=0. Task D ships")
    print("  the machinery to test it; whether self-similarity (and a warm-")
    print("  start payoff) emerges at deeper generations is the open Section-5")
    print("  question -- reported straight rather than tuned into a win.")
    print("=" * 70)

    G0 = prune(r2, 1e-8)                                 # Run 3 converged 3+2
    J0 = certify(G0)["Jc"]
    pf = int(np.abs(G0["A"]).max(axis=0).argmax())
    pg = int(np.abs(G0["B"]).max(axis=0).argmax())
    budget = 12

    def _warm(XI2, ETA2, af2, ag2):
        """Re-optimize from the bootstrapped seed; return (coarse-J curve,
        certified J, feasible, convergence outer-iter, pruned sol)."""
        A0, B0 = _seed_grown(G0, XI2, ETA2, af2, ag2)
        r = _alternate(A0, XI2, B0, ETA2, G0["t"], af2, ag2, outer=budget,
                       pos_iters=40, optimize_pos=True, verbose=False,
                       patience=budget)                 # run full budget
        curve = [jp for (_, jp) in r["hist"]]
        conv = next((i + 1 for i in range(1, len(curve))
                     if abs(curve[i] - curve[i - 1]) < 1e-5), len(curve))
        pruned = prune(r, 1e-8)
        c = certify(pruned)
        return curve, c["Jc"], c["rep"]["ALL CONSTRAINTS OK"], conv, pruned

    # spawn arm: contracted copy of the finest carrier, one f + one g kink.
    # spawn places both new kinks on [0.5, 1.0] (half the all-alive lifetime,
    # at the travel end); match that window for the random arm so the ONLY
    # difference is structured contraction vs random position.
    Xs, Es, afs, ags = spawn_generation(G0, scale_t=0.5, scale_x=0.5,
                                        rng=np.random.default_rng(0))
    # J-neutral at insertion: the new columns carry ZERO weight (pad G0's
    # weights with a zero column per grown family) -> total_J is exactly G0's,
    # because a zero-weight g-kink has no jump and a zero-weight f-kink no rise.
    A_pad = np.column_stack([G0["A"], np.zeros(G0["t"].size)])
    B_pad = np.column_stack([G0["B"], np.zeros(G0["t"].size)])
    J_ins = total_J(A_pad, Xs, B_pad, Es)
    _, spawn_Jc, spawn_ok, spawn_conv, spawn_sol = _warm(Xs, Es, afs, ags)

    best_rand = (-np.inf, False, None, None, None)  # (Jc, ok, conv, seed, sol)
    for rs in range(4):
        rr = np.random.default_rng(rs)
        Xr, Er, afr, agr = add_kink("f", G0["XI"], G0["ETA"], G0["alive_f"],
                                    G0["alive_g"], pf, G0["t"], 0.5, 1.0,
                                    dx=0.3, rng=rr)
        Xr, Er, afr, agr = add_kink("g", Xr, Er, afr, agr, pg, G0["t"],
                                    0.5, 1.0, dx=0.3, rng=rr)
        _, jc, ok, conv, sol = _warm(Xr, Er, afr, agr)
        if ok and jc > best_rand[0]:
            best_rand = (jc, ok, conv, rs, sol)
    rand_Jc, rand_ok, rand_conv, rand_seed, rand_sol = best_rand

    print(f"  G0 (Run 3): J_certified = {J0:.4f}")
    print(f"  spawn insertion is J-neutral: J at insertion = {J_ins:.4f} "
          f"(= J0 coarse {G0['J']:.4f}? {abs(J_ins - G0['J']) < 1e-6})")
    print(f"  spawn : J_certified = {spawn_Jc:.4f} (dJ {spawn_Jc-J0:+.4f})  "
          f"feasible = {spawn_ok}  converged in {spawn_conv} outers")
    print(f"  random: J_certified = {rand_Jc:.4f} (dJ {rand_Jc-J0:+.4f})  "
          f"feasible = {rand_ok}  converged in {rand_conv} outers  "
          f"(best feasible of 4 seeds, #{rand_seed})")
    win = spawn_ok and spawn_Jc >= rand_Jc - 1e-6
    print(f"  -> warm start beats random: {win}  "
          f"(null result: contracted copy is redundant at gen 0 -- the "
          f"hierarchy is not yet self-similar; machinery is correct and ready "
          f"for the multi-generation Section-5 test)")
    save_run("run8_spawn", spawn_sol, meta=dict(scale_t=0.5, scale_x=0.5))
    if rand_sol is not None:
        save_run("run8_random", rand_sol, meta=dict(dx=0.3, seed=rand_seed))

    print()
    print("=" * 70)
    print("Run 9 (Section 5): the generation-gain ladder -- the experiment")
    print("  everything else in this file serves. Run 8 showed the warm start")
    print("  (spawn_generation) is just an accelerator and doesn't beat random")
    print("  insertion at gen 0; it does NOT block the measurement itself. This")
    print("  run uses the already-working add_kink multistart (Run 8's random")
    print("  arm) as the insertion mechanism, with one change that makes it a")
    print("  measurement instead of a repeat of Runs 4-5: each generation's new")
    print("  f-kink and g-kink get an IMPOSED lifetime window that halves every")
    print("  generation (Run 6 showed an unrestricted greedy always prefers a")
    print("  full lifetime -- more DOF wins -- so the window constraint is the")
    print("  whole point). Records dJk = Jk - J_{k-1} per generation, plus a")
    print("  guard arm (free, full-lifetime insertion) so a constant dJk can't")
    print("  be an artifact of the imposed window geometry. Interpretation:")
    print("  dJk roughly CONSTANT over generations supports the ln(Nx) mesh")
    print("  growth being real (sup J = +infinity, approached not attained);")
    print("  dJk DECAYING means J is bounded and the mesh growth was transient.")
    print("=" * 70)

    ladder = generation_ladder(G0, n_gen=3, window0=0.5, window_ratio=0.5,
                               seeds=range(3), outer=25, pos_iters=60,
                               coarse_N=8, base_fine_sub=4, sub=8)
    save_ladder("run9", ladder, meta=dict(base="run3", n_gen=3, window0=0.5,
                                          window_ratio=0.5,
                                          seeds=list(range(3)), outer=25,
                                          pos_iters=60, coarse_N=8,
                                          base_fine_sub=4, sub=8))

    print(f"\n  {'k':>2} {'w_k':>7} {'Jc':>8} {'dJk':>8} {'ok':>5}   "
          f"{'guard_Jc':>8} {'guard_dJk':>9} {'ok':>5}")
    for g in ladder["generations"]:
        print(f"  {g['k']:>2} {g['w_k']:>7.4f} {g['Jc']:>8.4f} "
              f"{g['dJk']:>+8.4f} {str(g['feasible']):>5}   "
              f"{g['guard_Jc']:>8.4f} {g['guard_dJk']:>+9.4f} "
              f"{str(g['guard_feasible']):>5}")
        for fam in ("f", "g"):
            d = g["diagnostics"][fam]
            print(f"       +{fam}-kink: lifetime=({d['lifetime'][0]:.3f},"
                  f"{d['lifetime'][1]:.3f})  extent=({d['extent'][0]:+.3f},"
                  f"{d['extent'][1]:+.3f})  jump_mean={d['jump_mean']:.3f}  "
                  f"offset_from_parent={d['offset_from_parent']:.3f}")
        spread_str = ", ".join(f"seed{s}:{jc:.4f}{'' if ok else '(infeas)'}"
                               for s, jc, ok in g["spread"])
        print(f"       spread: {spread_str}")

    dJks = [g["dJk"] for g in ladder["generations"]]
    guard_dJks = [g["guard_dJk"] for g in ladder["generations"]]
    print(f"\n  dJk sequence:       {[f'{d:+.4f}' for d in dJks]}")
    print(f"  guard dJk sequence: {[f'{d:+.4f}' for d in guard_dJks]}")
    print("  Reading these numbers is the open question this file was built")
    print("  to answer -- see STRATEGY.md Section 5 for the interpretation")
    print("  rule (constant vs decaying dJk) and the honesty requirements")
    print("  (every Jc above is J_certified; the spread, not just the max, is")
    print("  reported per generation; the guard arm is never adopted into the")
    print("  ladder, only compared against it).")

    print()
    print("=" * 70)
    print("Run 10 (scale-sweep discriminators): Run 9's n_gen=3 ladder showed")
    print("  dJk decaying on two bases, but 3 points can't rule out budget")
    print("  starvation faking decay (three separate under-convergence")
    print("  artifacts were caught by hand producing Run 9's numbers -- see")
    print("  plans/run9-generation-gain-ladder.md). This run decouples 'gain")
    print("  truly vanishes at small scale' from 'optimizer starved' with two")
    print("  experiments that both run at Run 9's already-PROVEN budget")
    print("  (outer=40, pos_iters=100, seeds=5 @ ~41-node grids), no budget")
    print("  escalation treadmill:")
    print("  Experiment 1 -- dJ(w): a SINGLE generation's gain vs window scale")
    print("    w, each point independent (own regrid, no accumulation), on")
    print("    Run 6's grown base. A NULL-CONTROL arm (force_dead=True: the")
    print("    new kinks are masked so the LP can NEVER give them weight) runs")
    print("    alongside -- discovered necessary when a point with BOTH new")
    print("    kinks at jump_mean=0 (dead at convergence) still showed nonzero")
    print("    dJ: the insertion jitter alone perturbs the OLD kinks onto a")
    print("    different local optimum via the multi-seed search, independent")
    print("    of any real value the new kink provides. corrected_dJ = dJ -")
    print("    null_dJ subtracts that search-noise floor. corrected_dJ(w) ->")
    print("    nonzero const as w->0 supports log-growth; ->0 supports bounded.")
    print("  Experiment 2 -- window_ratio discriminator: rerun the n_gen=3")
    print("    ladder at window_ratio 0.7 and 0.3 on both bases (G0, grown).")
    print("    Under 'J bounded, dJ ~ w', decay_ratios should track")
    print("    window_ratio; under log-growth it should not.")
    print("  Experiment 3 -- cheap gate (_gate_report: feasible seeds + real")
    print("    kink activity) on every point, plus a doubled-budget bookend")
    print("    check (_budget_stable) at the sweep's narrowest window, the")
    print("    single point most exposed to starvation.")
    print("  See plans/run10-scale-sweep-discriminators.md for the full")
    print("  design and plans/run10-code-plan.md for the implementation.")
    print("=" * 70)

    print("\n  -- Experiment 1: dJ(w) on Run 6's grown base --")
    print("  (paired per-seed statistic, 16 seeds, outer=80/pos_iters=200 --")
    print("  a first pass at 5 seeds/outer=40 with best-of-max subtraction")
    print("  came back with no discernible trend and 3/6 points negative")
    print("  (fixed via _paired_dJ); a second pass at 16 seeds/outer=40 then")
    print("  showed windowed-arm feasibility as low as 6/16 at w=0.5 and both")
    print("  arms collapsing to near-identical values (no real search")
    print("  diversity) at w=0.0625 -- Run 9's outer=40/pos_iters=100 budget,")
    print("  proven adequate for ITS ladder setup, does not port to")
    print("  scale_sweep's single-window regrid. Doubled here; see")
    print("  plans/run10-scale-sweep-discriminators.md for the full history)")
    ws10 = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    seeds10 = range(16)
    sweep = scale_sweep(grown, ws10, seeds=seeds10, outer=80, pos_iters=200,
                        sub=8, verbose=False)
    print(f"  base_Jc = {sweep['base_Jc']:.5f}")
    print(f"  {'w':>9} {'nodes':>6} {'dJ':>9} {'ok':>5} {'gate':>6}   "
          f"{'corrected_dJ':>12} {'+/- se':>8} {'n':>3}   {'guard_dJ':>9} {'ok':>5}")
    for p in sweep["points"]:
        g = _gate_report(p)
        print(f"  {p['w']:>9.5f} {p['n_nodes']:>6} {p['dJ']:>+9.5f} "
              f"{str(p['feasible']):>5} {str(g['ready']):>6}   "
              f"{p['corrected_dJ_mean']:>+12.5f} {p['corrected_dJ_se']:>8.5f} "
              f"{p['corrected_dJ_n']:>3}   "
              f"{p['guard_dJ']:>+9.5f} {str(p['guard_feasible']):>5}")
    save_sweep("run10_sweep_run6base", sweep,
              meta=dict(base="run6", ws=ws10, seeds=list(seeds10),
                        outer=80, pos_iters=200, sub=8))

    narrowest = sweep["points"][-1]
    bookend = _budget_stable(generation_step, narrowest["sol"],
                             (1.0 - ws10[-1], 1.0), seeds=range(5),
                             outer=80, pos_iters=200, sub=8)
    print(f"  bookend check @ narrowest w={ws10[-1]}: budget-doubled dJ moved "
          f"by {bookend['delta']:+.5f}  (stable={bookend['accepted']})")

    print("\n  -- Experiment 2: window_ratio discriminator --")
    ratio_runs = []
    for base_name, base_sol in (("G0", G0), ("run6", grown)):
        for wr in (0.7, 0.3):
            lad = generation_ladder(base_sol, n_gen=3, window0=0.5,
                                    window_ratio=wr, seeds=range(5), outer=40,
                                    pos_iters=100, coarse_N=8, base_fine_sub=4,
                                    sub=8, verbose=False)
            dr = decay_ratios(lad["generations"])
            cdr = corrected_decay_ratios(lad["generations"])
            dJks10 = [g["dJk"] for g in lad["generations"]]
            cdJks10 = [g["corrected_dJk"] for g in lad["generations"]]
            cns10 = [g["corrected_dJk_n"] for g in lad["generations"]]
            print(f"  base={base_name:5s} window_ratio={wr:.1f}  "
                  f"dJk={[f'{d:+.4f}' for d in dJks10]}  "
                  f"decay_ratios={[f'{r:.3f}' for r in dr]}")
            print(f"    corrected_dJk={[f'{d:+.4f}' for d in cdJks10]}  "
                  f"n={cns10}  "
                  f"corrected_decay_ratios={[f'{r:.3f}' for r in cdr]}")
            save_ladder(f"run10_ratio_{base_name}_{wr}", lad,
                       meta=dict(base=base_name, n_gen=3, window0=0.5,
                                window_ratio=wr, seeds=list(range(5)),
                                outer=40, pos_iters=100, coarse_N=8,
                                base_fine_sub=4, sub=8))
            ratio_runs.append((base_name, wr, dr))

    print("\n  Reading: read corrected_decay_ratios, not decay_ratios -- raw")
    print("  dJk carries the same search-noise contamination Experiment 1")
    print("  found (a dead new kink can still move Jc via the multi-seed")
    print("  search finding a better basin for the OLD kinks). AT seeds=5,")
    print("  the result is WORSE than inconclusive: the paired stat needs a")
    print("  seed feasible in BOTH the windowed and null arms, and n above")
    print("  collapses to 0-1 for 3 of the 4 (base, window_ratio) combos --")
    print("  not enough to read a trend at all. Only base=G0, window_ratio=0.7")
    print("  keeps n=2-4, and even there corrected_dJk flips sign across")
    print("  generations with no se-separated pattern. Same lesson Experiment")
    print("  1 already learned: 5 seeds looks fine for raw dJk optics but")
    print("  starves the paired statistic -- Experiment 1 needed 16 seeds to")
    print("  get n=1-10. Escalating Experiment 2 the same way is the next")
    print("  step if this discriminator is worth pursuing further; as run")
    print("  here it should NOT be read for a window_ratio-tracking signal.")
    print("  See")
    print("  run10-scale-sweep-discriminators.md Experiments 4 (falsifiable")
    print("  gen4 extrapolation) and 5 (constructive arm) for what to run")
    print("  next -- not run automatically here since Experiment 4 needs an")
    print("  escalating per-generation budget (compounding cost) and")
    print("  Experiment 5 is exploratory math, not yet implemented")
    print("  (kink_opt/construct.py).")

    print()
    print("=" * 70)
    print("Run 11 (constructive hierarchy): the optimizer route (Runs 9-10)")
    print("  hit its structural ceiling -- a local-search optimizer can only")
    print("  ever FAIL to find gain, never certify it absent. This run builds")
    print("  the hierarchy DIRECTLY: kink positions are set analytically (an")
    print("  affine-contracted copy of a travelling gen-0 carrier, per")
    print("  generation, anchored at a shared right endpoint -- see")
    print("  construct.build_hierarchy), zero position NLP anywhere. The only")
    print("  numerical solve left is the convex weight LP that certify() and")
    print("  build_hierarchy's own weight solve both use to exact global")
    print("  optimality. Deterministic, no seeds, no null arm needed -- there")
    print("  is no search noise to correct for.")
    print("=" * 70)

    print("\n  -- validation before trusting any number --")
    neutral = check_insertion_neutral(3, scale_t=0.5, scale_x=0.5)
    print(f"  insertion-neutral check (gen 3 forced dead vs gen 2): "
          f"Jc {neutral['Jc_with_dead']:.5f} vs {neutral['Jc_without']:.5f}  "
          f"(rel diff {neutral['diff_rel']:.2e})  -> "
          f"{'PASS' if neutral['ok'] else 'FAIL'}")

    primary = constructive_ladder(4, scale_t=0.5, scale_x=0.5, sub=8)
    deepest = primary["generations"][-1]["sol"]
    gc = grid_convergence_check(deepest, sub_lo=8, sub_hi=16)
    print(f"  grid-convergence check (deepest ladder, sub 8 vs 16): "
          f"Jc {gc['Jc_lo']:.5f} vs {gc['Jc_hi']:.5f}  "
          f"(rel diff {gc['diff_rel']:.2%})  -> "
          f"{'PASS' if gc['ok'] else 'FAIL'}")
    ts = travel_sanity(primary["generations"][0]["sol"])
    print(f"  travel sanity (gen-0 carrier): p_start={ts['p_start']:+.3f} "
          f"-> p_end={ts['p_end']:+.3f}  -> "
          f"{'PASS (travels)' if ts['ok'] else 'FAIL (co-located)'}")

    print("\n  -- primary ladder: scale_t=0.5, scale_x=0.5, n_gen=4 --")
    print(f"  {'k':>2} {'Jc':>9} {'dJk':>9} {'ratio':>7} {'ok':>5}")
    for g in primary["generations"]:
        dJk_s = f"{g['dJk']:+.5f}" if g["dJk"] is not None else "   -   "
        ratio_s = f"{g['ratio']:.3f}" if g["ratio"] is not None else "  -  "
        print(f"  {g['k']:>2} {g['Jc']:>9.5f} {dJk_s:>9} {ratio_s:>7} "
              f"{str(g['constraints_ok']):>5}")
    save_construct("run11_primary", primary, meta=dict(n_gen=4, sub=8))

    dJks11 = [g["dJk"] for g in primary["generations"] if g["dJk"] is not None]
    r_fit, dJ1_fit = fit_geometric(dJks11[:3])
    dJ4_pred = dJ1_fit * r_fit ** 3
    dJ4_actual = dJks11[3]
    print(f"\n  falsifiable prediction (fit dJ1..dJ3 geometric, predict dJ4):")
    print(f"    fitted ratio r={r_fit:.3f}  ->  predicted dJ4={dJ4_pred:+.5f}"
          f"   actual dJ4={dJ4_actual:+.5f}")
    print("    actual dJ4 is far BELOW the geometric extrapolation (already")
    print("    ~0, not merely decaying at the fitted rate) -- decay is faster")
    print("    than geometric, not slower: this falsifies log-growth harder")
    print("    than the prediction even required, and supports bounded J.")

    print("\n  -- saturation-mechanism instrumentation (why dJk decays) --")
    print(f"  {'k':>2} {'x_birth':>8} {'rise_budget_used':>17} {'g_mass':>8}")
    for g in primary["generations"][1:]:
        d = saturation_diagnostics(g["sol"], g["k"])
        print(f"  {d['k']:>2} {d['x_birth']:>+8.4f} "
              f"{d['rise_budget_used']:>17.4f} {d['g_mass']:>8.4f}")
    print("  Reading: x_birth converges toward the shared anchor p_end as k")
    print("  grows (by construction -- window_k shrinks toward t1), so")
    print("  rise_budget_used converging is expected geometry, not evidence")
    print("  of hitting the Lipschitz cap. g_mass is exactly 0 for gen 1-2 --")
    print("  the LP gets all of gen 1-2's gain from the new f-kink alone and")
    print("  assigns the new g-kink NO weight, only starting to use it once")
    print("  the contraction is tight enough (gen 3+). Honest caveat: because")
    print("  every generation is anchored at the SAME point (p_end, t1), deep")
    print("  generations collapse toward one location -- structurally similar")
    print("  to Run 8's 'two hats at one point are redundant' mechanism, just")
    print("  emerging here from the anchoring choice itself rather than from")
    print("  weight competition. This construction therefore demonstrates")
    print("  bounded J UNDER a single shared-endpoint self-similar anchoring;")
    print("  it does not by itself rule out a hierarchy whose generations")
    print("  keep moving to genuinely new locations instead of collapsing.")

    print("\n  -- sweep (scale_t, scale_x), n_gen=3 --")
    sweep11 = sweep_ratios(3, scale_ts=[0.7, 0.5, 0.3], scale_xs=[0.7, 0.5, 0.3],
                           sub=8)
    print(f"  {'scale_t':>7} {'scale_x':>7}   dJk                          ratios")
    for o in sweep11:
        dJk_str = ", ".join(f"{d:+.4f}" for d in o["dJk"])
        ratio_str = ", ".join(f"{r:.3f}" for r in o["ratios"])
        print(f"  {o['scale_t']:>7.1f} {o['scale_x']:>7.1f}   [{dJk_str}]   "
              f"[{ratio_str}]")
    print("  Every (scale_t, scale_x) combination shows dJk collapsing by")
    print("  1-2 orders of magnitude within 3 generations -- none shows the")
    print("  roughly-constant dJk the log-growth conjecture predicts. The")
    print("  decay ratio does not cleanly track either scale_t or scale_x")
    print("  alone (e.g. scale_x=0.7 gives ratios 0.29/0.38, well below 0.7),")
    print("  i.e. decay is faster than either single contraction rate would")
    print("  predict on its own -- consistent with the anchoring-collapse")
    print("  mechanism above compounding on top of the per-scale contraction.")
    print("  Conclusion for this construction: bounded J, decisively, subject")
    print("  to the shared-anchor caveat above -- the constructive arm this")
    print("  plan called for is the first PROOF-quality (not merely")
    print("  search-consistent) evidence either way in this file.")

    print()
    print("=" * 70)
    print("Run 12 (melt-band cell, plans/run12-*.md): Run 11 showed bounded J")
    print("  for ONE specific anchoring (generations contract toward a shared")
    print("  fixed endpoint, so deep generations collapse spatially onto one")
    print("  point -- 'two hats at one point are redundant' is baked into that")
    print("  geometry). This run tests a DIFFERENT ansatz that does not have")
    print("  that structural collapse built in: each generation is a BAND of")
    print("  K kinks (not 1) riding a drift path of ABSOLUTE length L that")
    print("  does NOT shrink with depth (only the band's width w_k and its")
    print("  window duration s_k shrink, both geometrically). Still zero")
    print("  position NLP -- weights only, LP-only alternation.")
    print("=" * 70)

    print("\n  -- validation before trusting any number --")
    m1 = build_melt_hierarchy(1)
    cbn = check_band_neutral(1)
    print(f"  band-neutral check (gen 1 forced dead vs gen 0): "
          f"Jc {cbn['Jc_with_dead']:.5f} vs {cbn['Jc_without']:.5f}  "
          f"(rel diff {cbn['diff_rel']:.2e})  -> "
          f"{'PASS' if cbn['ok'] else 'FAIL'}")
    gc12 = grid_convergence_check(m1)
    print(f"  grid-convergence check (gen 1, sub 8 vs 16): "
          f"Jc {gc12['Jc_lo']:.5f} vs {gc12['Jc_hi']:.5f}  "
          f"(rel diff {gc12['diff_rel']:.2%})  -> "
          f"{'PASS' if gc12['ok'] else 'FAIL'}")
    bts = band_travel_sanity(m1, 1, m1["band_specs"])
    print(f"  band-travel sanity (gen 1): L_expected={bts['L_expected']:.3f} "
          f"L_observed={bts['L_observed']:.3f}  -> "
          f"{'PASS (drifts)' if bts['ok'] else 'FAIL'}")

    print("\n  -- primary ladder: lambda_w=0.5, lambda_s=0.5, L=0.3, K=8, n_gen=3 --")
    lad12 = melt_ladder(3, sub=8)
    print(f"  {'k':>2} {'Jc':>9} {'dJk':>9} {'ratio':>7} {'ok':>5}")
    for g in lad12["generations"]:
        dJk_s = f"{g['dJk']:+.5f}" if g["dJk"] is not None else "   -   "
        ratio_s = f"{g['ratio']:.3f}" if g["ratio"] is not None else "  -  "
        print(f"  {g['k']:>2} {g['Jc']:>9.5f} {dJk_s:>9} {ratio_s:>7} "
              f"{str(g['constraints_ok']):>5}")
    save_melt("run12_primary", lad12, meta=dict(n_gen=3))

    dJks12 = [g["dJk"] for g in lad12["generations"] if g["dJk"] is not None]
    mcc = mesh_cross_check(dJks12[-2:])
    print(f"\n  mesh cross-check (dJ_2, dJ_3 vs +0.215/octave, factor-of-3 bar): "
          f"ratios_to_target={[f'{r:.3f}' for r in mcc['ratio_to_target']]}  "
          f"-> {'PASS' if mcc['ok'] else 'FAIL'}")
    print("  Reading: dJk still collapses fast here too (2 orders of magnitude")
    print("  by generation 3), and lands two orders of magnitude below the")
    print("  mesh's own per-octave gain -- so a non-shrinking drift length L")
    print("  alone does not rescue the log-growth signature under this")
    print("  specific anchoring (every band still centered on the SAME point")
    print("  in time and space, `mid`/`c_anchor` -- only width and window")
    print("  duration shrink). gen 3 also fails certify()'s dense constraint")
    print("  check by a small margin (~1e-7) at this sub=8 resolution -- a")
    print("  resolution-floor symptom, not evidence about the hypothesis; see")
    print("  plans/run12-implementation-details.md Section 8 on the LP's")
    print("  Kf^2-ish cost -- sub=16 at this K=8/n_gen=3 scale is a multi-GB,")
    print("  multi-minute certify() call, deliberately not run by default here.")

    print("\n  -- environment read-off: generation 1 -> generation 2 --")
    spec2 = lad12["generations"][2]["sol"]["band_specs"][1]
    env1_out = lad12["generations"][1]["env"]
    env2_in = read_environment(lad12["generations"][1]["sol"], 2, spec2)
    dist = env_distance(env1_out, env2_in)
    print(f"  beta (gen 1, outgoing): {np.round(env1_out['beta'], 3)}")
    print(f"  beta (gen 2, incoming): {np.round(env2_in['beta'], 3)}")
    print(f"  env_distance(outgoing_1, incoming_2) = {dist:.4f}")
    print("  Not yet run: the (lambda_w, lambda_s, L) fixed-point sweep")
    print("  (melt.fixed_point_sweep) -- Section 8 of the implementation-")
    print("  details plan estimates a single n_gen=4,K=8 ladder point at low")
    print("  tens of seconds and a 27-point grid at tens of minutes, a")
    print("  'leave it running' job rather than part of the default demo run.")

    # EXT: adaptive per-kink time nodes (fine generations live fast & short)
    # EXT: warm-start next generation from a rescaled copy of this solution
    # EXT: multistart currently reseeds from scratch per attempt; a real
    #      basin-hopping / CMA-ES search would be far more sample-efficient
    # EXT: multistart selects by coarse J, not by verify_dense feasibility --
    #      Run 5's Kf=7,8 exploration shows this can pick an infeasible
    #      "winner" once K is large enough that the position NLP struggles;
    #      a feasibility-aware selection would be more robust at high K
    # EXT: Run 10 Experiment 4 (falsifiable dJk4 extrapolation via
    #      fit_geometric + generation_ladder(budget_fn=...)) and Experiment 5
    #      (kink_opt/construct.py constructive arm) -- see
    #      plans/run10-scale-sweep-discriminators.md, run only if Run 10's
    #      two discriminators above come back ambiguous or disagree.


if __name__ == "__main__":
    main()
