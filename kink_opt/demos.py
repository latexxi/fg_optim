"""Run 1-9: narrated demos. Not a test suite -- each run's configuration and
commentary explains what the previous run left on the table and why the next
one's hyperparameters were chosen. See CLAUDE.md's "The __main__ block"."""

import numpy as np

from .lp import lp_weights_f, lp_weights_g
from .objective import total_J
from .solver import run, multistart, _alternate
from .verify import certify, report, n_live_nodes, graded_grid, _ub
from .topology import (add_kink, spawn_generation, _seed_grown, prune,
                       grow_topology, generation_ladder)


def main():
    print("=" * 70)
    print("Run 1 (sanity): single co-located static kink pair, positions FROZEN")
    print("        exact discrete optimum is J = 2 (tent, bang-bang schedule)")
    print("=" * 70)
    r0 = run(N=24, Kf=1, Kg=1, outer=3, seed="static",
             optimize_pos=False, verbose=True)
    report("static 1+1", r0)

    print()
    print("=" * 70)
    print("Run 2: same single kink pair, positions FREE (can travel discover >2?)")
    print("=" * 70)
    r1 = run(N=16, Kf=1, Kg=1, outer=6, seed="static", pos_iters=50)
    report("travel 1+1", r1)

    print()
    print("=" * 70)
    print("Run 3: Kf=3, Kg=2, positions free, seeded at center")
    print("=" * 70)
    r2 = run(N=16, Kf=3, Kg=2, outer=40, seed="static", pos_iters=40, patience=5)
    report("static 3+2", r2)

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
        certified J, feasible, convergence outer-iter)."""
        A0, B0 = _seed_grown(G0, XI2, ETA2, af2, ag2)
        r = _alternate(A0, XI2, B0, ETA2, G0["t"], af2, ag2, outer=budget,
                       pos_iters=40, optimize_pos=True, verbose=False,
                       patience=budget)                 # run full budget
        curve = [jp for (_, jp) in r["hist"]]
        conv = next((i + 1 for i in range(1, len(curve))
                     if abs(curve[i] - curve[i - 1]) < 1e-5), len(curve))
        c = certify(prune(r, 1e-8))
        return curve, c["Jc"], c["rep"]["ALL CONSTRAINTS OK"], conv

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
    _, spawn_Jc, spawn_ok, spawn_conv = _warm(Xs, Es, afs, ags)

    best_rand = (-np.inf, False, None, None)            # (Jc, ok, conv, seed)
    for rs in range(4):
        rr = np.random.default_rng(rs)
        Xr, Er, afr, agr = add_kink("f", G0["XI"], G0["ETA"], G0["alive_f"],
                                    G0["alive_g"], pf, G0["t"], 0.5, 1.0,
                                    dx=0.3, rng=rr)
        Xr, Er, afr, agr = add_kink("g", Xr, Er, afr, agr, pg, G0["t"],
                                    0.5, 1.0, dx=0.3, rng=rr)
        _, jc, ok, conv = _warm(Xr, Er, afr, agr)
        if ok and jc > best_rand[0]:
            best_rand = (jc, ok, conv, rs)
    rand_Jc, rand_ok, rand_conv, rand_seed = best_rand

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

    # EXT: adaptive per-kink time nodes (fine generations live fast & short)
    # EXT: warm-start next generation from a rescaled copy of this solution
    # EXT: multistart currently reseeds from scratch per attempt; a real
    #      basin-hopping / CMA-ES search would be far more sample-efficient
    # EXT: multistart selects by coarse J, not by verify_dense feasibility --
    #      Run 5's Kf=7,8 exploration shows this can pick an infeasible
    #      "winner" once K is large enough that the position NLP struggles;
    #      a feasibility-aware selection would be more robust at high K


if __name__ == "__main__":
    main()
