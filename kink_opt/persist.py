"""Save/load run setups + results under ./results/<tag>/, so later analysis
(diagnostics, dJk trends, f/g visual inspection) works from disk instead of
rerunning the optimization. Every "sol" dict in this codebase (run(),
_alternate(), grow_topology(), generation_step()'s sol) shares one shape:
A, XI, B, ETA, t, J, hist, alive_f, alive_g -- save_sol/load_sol covers all
of them, plus the dense evaluated f(x,t)/g(x,t) fields and a heatmap figure."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .viz import field_grid, plot_heatmaps


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_sol(path, sol, nx=401):
    """Save one solution dict to <path>.npz -- kink coordinates (A, XI, B,
    ETA, t, alive_f, alive_g), J, hist, and the dense evaluated f(x,t),
    g(x,t) fields (x, F, G)."""
    x, F, G = field_grid(sol, nx=nx)
    np.savez(Path(path).with_suffix(".npz"),
             A=sol["A"], XI=sol["XI"], B=sol["B"], ETA=sol["ETA"], t=sol["t"],
             J=sol["J"], hist=np.array(sol.get("hist", []), dtype=float),
             alive_f=sol["alive_f"], alive_g=sol["alive_g"],
             x=x, F=F, G=G)


def load_sol(path):
    """-> dict, same shape run()/_alternate() return (plus extra x/F/G dense
    field keys, harmless to certify()/generation_step()/etc, which only read
    A, XI, B, ETA, t)."""
    z = np.load(Path(path).with_suffix(".npz"))
    return dict(A=z["A"], XI=z["XI"], B=z["B"], ETA=z["ETA"], t=z["t"],
                J=float(z["J"]), hist=[tuple(row) for row in z["hist"]],
                alive_f=z["alive_f"], alive_g=z["alive_g"],
                x=z["x"], F=z["F"], G=z["G"])


def save_figure(path, sol, nx=401):
    fig = plot_heatmaps(sol, nx=nx)
    fig.savefig(Path(path).with_suffix(".png"), dpi=100)
    plt.close(fig)


def save_run(tag, sol, meta=None, root="results", nx=401):
    """Save one solution dict + its provenance under results/<tag>/."""
    d = Path(root) / tag
    d.mkdir(parents=True, exist_ok=True)
    save_sol(d / "sol", sol, nx=nx)
    save_figure(d / "fields", sol, nx=nx)
    info = dict(tag=tag, timestamp=datetime.now(timezone.utc).isoformat(),
                git_sha=_git_sha(), J=float(sol["J"]), **(meta or {}))
    (d / "meta.json").write_text(json.dumps(_jsonable(info), indent=2))


def load_run(tag, root="results"):
    """-> (sol, meta). sol plugs directly into certify()/report()/
    generation_step()/etc; meta is the saved kwargs+provenance dict."""
    d = Path(root) / tag
    sol = load_sol(d / "sol")
    meta = json.loads((d / "meta.json").read_text())
    return sol, meta


def save_ladder(tag, ladder, meta=None, root="results", nx=401):
    """Save a generation_ladder() result under results/<tag>/:
      results/<tag>/gen{k}/sol.npz, fields.png -- full solution + heatmap
                                                   AFTER generation k
      results/<tag>/ladder.json               -- {base_Jc, generations, meta}
    (generations[] entries have "sol" stripped -- it lives in gen{k}/sol.npz)."""
    d = Path(root) / tag
    d.mkdir(parents=True, exist_ok=True)
    generations = []
    for g in ladder["generations"]:
        k = g["k"]
        gd = d / f"gen{k}"
        gd.mkdir(parents=True, exist_ok=True)
        save_sol(gd / "sol", g["sol"], nx=nx)
        save_figure(gd / "fields", g["sol"], nx=nx)
        generations.append(_jsonable({key: v for key, v in g.items() if key != "sol"}))
    info = dict(tag=tag, timestamp=datetime.now(timezone.utc).isoformat(),
                git_sha=_git_sha(), base_Jc=float(ladder["base_Jc"]),
                generations=generations, **(meta or {}))
    (d / "ladder.json").write_text(json.dumps(info, indent=2))


def load_ladder(tag, root="results"):
    """-> dict(generations=[...], base_Jc=..., meta=...), structurally the
    same as generation_ladder()'s return (each generation dict regains its
    "sol" key, loaded from gen{k}/sol.npz) -- so a loaded ladder can feed
    straight into another generation_step() call to extend it without
    rerunning generations 1..k."""
    d = Path(root) / tag
    info = json.loads((d / "ladder.json").read_text())
    generations = info.pop("generations")
    for g in generations:
        g["sol"] = load_sol(d / f"gen{g['k']}" / "sol")
    return dict(generations=generations, base_Jc=info.pop("base_Jc"), meta=info)


def save_sweep(tag, sweep, meta=None, root="results", nx=401):
    """Save a scale_sweep() result (Run 10 Experiment 1) under
    results/<tag>/:
      results/<tag>/w{i}/sol.npz, fields.png -- full solution + heatmap AT
                                                 that sweep point (index i,
                                                 in the order scale_sweep
                                                 was given `ws`, not sorted)
      results/<tag>/sweep.json               -- {base_Jc, points, meta}
    (points[] entries have "sol" stripped -- it lives in w{i}/sol.npz), same
    layout convention as save_ladder's gen{k}/."""
    d = Path(root) / tag
    d.mkdir(parents=True, exist_ok=True)
    points = []
    for i, p in enumerate(sweep["points"]):
        pd = d / f"w{i}"
        pd.mkdir(parents=True, exist_ok=True)
        save_sol(pd / "sol", p["sol"], nx=nx)
        save_figure(pd / "fields", p["sol"], nx=nx)
        points.append(_jsonable({key: v for key, v in p.items() if key != "sol"}))
    info = dict(tag=tag, timestamp=datetime.now(timezone.utc).isoformat(),
                git_sha=_git_sha(), base_Jc=float(sweep["base_Jc"]),
                points=points, **(meta or {}))
    (d / "sweep.json").write_text(json.dumps(info, indent=2))


def load_sweep(tag, root="results"):
    """-> dict(points=[...], base_Jc=..., meta=...), structurally the same as
    scale_sweep()'s return (each point dict regains its "sol" key, loaded
    from w{i}/sol.npz)."""
    d = Path(root) / tag
    info = json.loads((d / "sweep.json").read_text())
    points = info.pop("points")
    for i, p in enumerate(points):
        p["sol"] = load_sol(d / f"w{i}" / "sol")
    return dict(points=points, base_Jc=info.pop("base_Jc"), meta=info)
