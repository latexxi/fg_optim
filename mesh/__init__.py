"""mesh — the fg_opt3 mesh-grid optimizer, relevant parts copied into fg_opt4.

This is the FULL-MESH solver (every (x,t) node is a free variable, solved by exact
alternating LPs), distinct from the sibling `kink_opt` package (hat-basis / kink
coordinates). It exists here to build the ADAPTIVE, harvest-gauge refinement
described in plans/mesh/ — a tau-gauge time grid + band-only x refinement, warm
started by regauge+prolong — on top of the uniform dyadic baseline.

Public API (baseline, copied verbatim from fg_opt3):
  make_grids                              uniform dyadic grid
  build_constraints, check_feasible, idx  LP constraints (non-uniform-x ready)
  compute_J, harvest_per_interval         objective + its tau-gauge CDF material
  build_c_f, build_c_g,
  solve_f_given_g, solve_g_given_f        the two convex sub-solves
  HiGHSWarmLP                             warm-started persistent LP
  alternating_maximization                the block-coordinate driver
  interpolate_to_next_level,
  dyadic_refinement                       UNIFORM baseline refinement (comparison)

Adaptive modules (adapt.py, prolong.py, refine_adapt.py) are added by the tasks in
plans/mesh/ and re-exported here as they land.
"""
from .grid import make_grids
from .constraints import build_constraints, check_feasible, idx
from .objective import compute_J, harvest_per_interval
from .lp_subproblem import build_c_f, build_c_g, solve_f_given_g, solve_g_given_f
from .highs_warm import HiGHSWarmLP
from .alternating import alternating_maximization
from .refine_baseline import interpolate_to_next_level, dyadic_refinement
from .adapt import BAND, tau_regrid, band_refine
from .prolong import regauge_time, prolong_x, adaptive_warm_start
from .refine_adapt import adaptive_refinement, two_d_climb

__all__ = [
    "make_grids", "build_constraints", "check_feasible", "idx",
    "compute_J", "harvest_per_interval",
    "build_c_f", "build_c_g", "solve_f_given_g", "solve_g_given_f",
    "HiGHSWarmLP", "alternating_maximization",
    "interpolate_to_next_level", "dyadic_refinement",
    # adaptive harvest-gauge refinement (plans/mesh/)
    "BAND", "tau_regrid", "band_refine",
    "regauge_time", "prolong_x", "adaptive_warm_start",
    "adaptive_refinement", "two_d_climb",
]
