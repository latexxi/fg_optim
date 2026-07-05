"""
kink_opt -- Prototype: kink-coordinate optimization for

    max  J[f,g] = int_0^1 int_{-1}^1 f_t(x,t) * g_xx(x,t) dx dt

Representation (negative-hat basis)
-----------------------------------
    f(x,t) = - sum_i a_i(t) * hat(x; xi_i(t)),    a_i >= 0
    g(x,t) = - sum_m b_m(t) * hat(x; eta_m(t)),   b_m >= 0

where hat(x; c) is the piecewise-linear "tent" with hat(+-1)=0, hat(c)=1.

Constraint dictionary in these coordinates:
    convexity in x       <=>  a_i >= 0, b_m >= 0                       (exact)
    Lipschitz |f_x|<=1   <=>  sum_i a_i/(1+xi_i) <= 1  and
                              sum_i a_i/(1-xi_i) <= 1  per time node   (exact)
    boundary f(+-1,t)=0  <=>  built into the basis                     (exact)
    terminal f(x,1)=0    <=>  a_i(t_N) = 0
    initial  g(x,0)=0    <=>  b_m(t_0) = 0
    f_t >= 0             <=>  f(.,t_{k+1}) - f(.,t_k) >= 0 at the UNION
                              of kink positions of both slices.
                              (difference of two convex PL functions is PL
                              with kinks exactly at that union, and it
                              vanishes at x=+-1, so this check is EXACT)
    g_t <= 0             <=>  analogous, with reversed sign.

Objective (harvest form, no near-singular quadrature):
    g_xx(.,t) = sum_m j_m(t) delta(x - eta_m(t)),  j_m = 2 b_m / (1 - eta_m^2)
    J = sum_m int_0^1 j_m(t) * f_t(eta_m(t), t) dt
Discretized with midpoint rule per time step:
    J ~= sum_k sum_m j_m^{k+1/2} * [ f(eta^{k+1/2}, t_{k+1}) - f(eta^{k+1/2}, t_k) ]
(the dt cancels: J is literally "rise of f harvested at g's kinks").

Optimization strategy (exploits partial convexity)
--------------------------------------------------
    1. positions frozen  -> J is LINEAR in the f-weights  a  -> LP (HiGHS)
    2. positions frozen  -> J is LINEAR in the g-weights  b  -> LP (HiGHS)
    3. weights frozen    -> smooth-ish NLP in positions (L-BFGS-B with
                            penalties for ordering / monotonicity / Lipschitz,
                            analytic gradients -- see grad_total_J and
                            grad_penalty), then re-run 1-2 to restore exact
                            feasibility.
Every LP block is solved to global optimality; only step 3 is nonconvex.

This is a PROTOTYPE: small K, modest N.
Extension points are marked with  # EXT (see demos.py).

Package layout
--------------
    geometry   -- hat_matrix, conv_eval, weight-LP box bounds
    lp         -- lp_weights_f / lp_weights_g (the convex weight blocks)
    objective  -- total_J, analytic gradients, penalty, optimize_positions
    verify     -- refine_time, graded_grid, verify_dense, certify, report
    solver     -- _alternate, run, multistart (the block-coordinate driver)
    topology   -- add_kink, prune, spawn_generation, grow_topology,
                  generation_ladder (Task B/D + the Run 9 measurement)
    construct  -- build_hierarchy, constructive_ladder (Run 11 -- the
                  constructive, optimizer-free self-similar hierarchy)
    melt       -- build_melt_hierarchy, melt_ladder (Run 12 -- the melt-band
                  cell construction: K kinks per generation riding a
                  non-shrinking drift length L, plus the environment
                  read-off/fixed-point machinery)
    demos      -- the narrated Run 1-12 __main__ sequence
"""

from .geometry import MARGIN, GAP, PEN_W, hat_matrix, conv_eval
from .lp import lp_weights_f, lp_weights_g
from .objective import (total_J, grad_total_J, penalty, grad_penalty,
                        optimize_positions)
from .verify import (refine_time, graded_grid, n_live_nodes, verify_dense,
                     certify, report)
from .solver import run, multistart
from .topology import (add_kink, spawn_generation, prune, generation_step,
                       generation_ladder, grow_topology)
from .construct import (build_hierarchy, constructive_ladder, sweep_ratios,
                        check_insertion_neutral, grid_convergence_check,
                        travel_sanity, saturation_diagnostics)
from .melt import (build_band, build_melt_hierarchy, melt_ladder, melt_sweep,
                   read_environment, env_distance, check_band_neutral,
                   band_travel_sanity, mesh_cross_check, fixed_point_sweep)

__all__ = [
    "MARGIN", "GAP", "PEN_W", "hat_matrix", "conv_eval",
    "lp_weights_f", "lp_weights_g",
    "total_J", "grad_total_J", "penalty", "grad_penalty", "optimize_positions",
    "refine_time", "graded_grid", "n_live_nodes", "verify_dense", "certify",
    "report",
    "run", "multistart",
    "add_kink", "spawn_generation", "prune", "generation_step",
    "generation_ladder", "grow_topology",
    "build_hierarchy", "constructive_ladder", "sweep_ratios",
    "check_insertion_neutral", "grid_convergence_check", "travel_sanity",
    "saturation_diagnostics",
    "build_band", "build_melt_hierarchy", "melt_ladder", "melt_sweep",
    "read_environment", "env_distance", "check_band_neutral",
    "band_travel_sanity", "mesh_cross_check", "fixed_point_sweep",
]
