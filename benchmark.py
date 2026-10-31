"""Run every solver on the SAME 330 targets and save results.

    python benchmark.py              # all solvers, both modes
    python benchmark.py --solvers ikpy mink --multistart 5

Modes
  single      : one attempt starting from HOME (what a robot would do with no prior knowledge)
  multistart  : HOME first, then random starts inside the joint limits, stop at first success
Outputs: results.csv (every solve), summary.csv (table for the report), plots/*.png
"""
import sys, argparse, importlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "solvers"))
import numpy as np, csv
from common import *

REGISTRY = {"ikpy": ("ikpy_solver", "IKPySolver"), "mink": ("mink_solver", "MinkSolver"),
            "pinocchio": ("pinocchio_solver", "PinocchioSolver"), "pink": ("pink_solver", "PinkSolver"),
            "placo": ("placo_solver", "PlacoSolver")}       # placo: added later


def load_solver(key):
    mod, cls = REGISTRY[key]
    try:
        return getattr(importlib.import_module(mod), cls)()
    except Exception as e:                                   # missing package / file -> skip, don't crash
        print(f"[skip] {key}: {type(e).__name__}: {e}"); return None


def attempt_starts(t, kin, n_extra):
    rng = np.random.default_rng(1000 + t.id)                 # same random starts for every solver
    return [HOME_Q] + [rng.uniform(kin.lo, kin.hi) for _ in range(n_extra)]


def run_target(solver, kin, t, starts):
    best, total_t = None, 0.0
    for k, q0 in enumerate(starts, 1):
        res = solver.solve(t, q0); total_t += res.solve_time_s
        m = evaluate(kin, t, res.q); m["attempts"] = k
        if best is None or m["pos_err_mm"] + m["ang_err_deg"] < best["pos_err_mm"] + best["ang_err_deg"]: best = m
        if m["success"]: best = m; break
    best["time_ms"] = total_t * 1000; best["attempts"] = k
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="so101", choices=list(ROBOT_REGISTRY.keys()),
                    help="Target robot model (so101, piper, ur5, franka)")
    ap.add_argument("--solvers", nargs="+", default=["ikpy", "mink", "pinocchio", "pink"])
    ap.add_argument("--multistart", type=int, default=5, help="max attempts in multistart mode")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N targets")
    a = ap.parse_args()

    robot_cfg = get_robot_config(a.robot)
    kin = Kin(robot_config=robot_cfg)
    targets = load_targets()
    if a.limit: targets = targets[:a.limit]
    rows = []
    for key in a.solvers:
        s = load_solver(key)
        if s is None: continue
        s.setup(kin); s.solve(targets[0], HOME_Q)               # warm-up (first call is slow)
        for mode, n_extra in (("single", 0), ("multistart", a.multistart - 1)):
            t0 = time.time()
            for t in targets:
                m = run_target(s, kin, t, attempt_starts(t, kin, n_extra))
                rows.append(dict(solver=s.name, mode=mode, target=t.id, category=t.category,
                                 safe_for_real=int(t.safe_for_real), **m))
            print(f"{s.name:<10} {mode:<11} done in {time.time()-t0:5.1f} s")

    keys = ["solver", "mode", "target", "category", "safe_for_real", "success", "pos_err_mm", "ang_err_deg",
            "limit_violation_rad", "time_ms", "attempts"]
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    summarize(rows)


def summarize(rows):
    import collections
    groups = collections.defaultdict(list)
    for r in rows: groups[(r["solver"], r["mode"], r["category"])].append(r)
    out = []
    for (s, mo, c), g in sorted(groups.items()):
        tm = np.array([x["time_ms"] for x in g]); ok = [x for x in g if x["success"]]
        out.append(dict(solver=s, mode=mo, category=c, n=len(g), success_pct=round(100 * len(ok) / len(g), 1),
                        median_time_ms=round(float(np.median(tm)), 2), p95_time_ms=round(float(np.percentile(tm, 95)), 2),
                        median_pos_err_mm=round(float(np.median([x["pos_err_mm"] for x in g])), 3),
                        mean_attempts=round(float(np.mean([x["attempts"] for x in g])), 2),
                        limit_violations=sum(x["limit_violation_rad"] > LIMIT_TOL for x in g)))
    with open("summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
    print("\n%-10s %-11s %-16s %4s %8s %10s %9s %9s" % ("solver", "mode", "category", "n", "succ %", "med ms", "p95 ms", "attempts"))
    for o in out:
        print("%-10s %-11s %-16s %4d %8.1f %10.2f %9.2f %9.2f" % (o["solver"], o["mode"], o["category"], o["n"],
              o["success_pct"], o["median_time_ms"], o["p95_time_ms"], o["mean_attempts"]))
    try: plots(out, rows)
    except ImportError: print("\n(pip install matplotlib to get the plots)")


def plots(summ, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Path("plots").mkdir(exist_ok=True)
    solvers = sorted({o["solver"] for o in summ}); cats = ["reachable", "near_limit"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axs, ("single", "multistart")):
        for i, c in enumerate(cats):
            v = [next((o["success_pct"] for o in summ if o["solver"] == s and o["mode"] == mode and o["category"] == c), 0) for s in solvers]
            ax.bar(np.arange(len(solvers)) + i * 0.38, v, 0.38, label=c)
        ax.set_xticks(np.arange(len(solvers)) + 0.19); ax.set_xticklabels(solvers); ax.set_title(f"Success rate - {mode}")
        ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("success %"); axs[0].legend(); plt.tight_layout(); plt.savefig("plots/success_rate.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[r["time_ms"] for r in rows if r["solver"] == s and r["mode"] == "single" and r["category"] == "reachable"] for s in solvers]
    ax.boxplot(data); ax.set_xticklabels(solvers); ax.set_ylabel("solve time (ms), reachable, single start"); ax.set_yscale("log"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("plots/solve_time.png", dpi=150); plt.close()
    print("\nsaved results.csv, summary.csv, plots/success_rate.png, plots/solve_time.png")


if __name__ == "__main__":
    main()
