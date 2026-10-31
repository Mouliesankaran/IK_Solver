"""Scorecard from results.csv:  python compare.py
Prints the 4-question comparison, the best solver per use case, and the targets nobody could solve.
Also saves scorecard.csv."""
import pandas as pd

r = pd.read_csv("results.csv")
def pct(df): return round(100 * df.success.mean(), 1)

rows = []
for s, g in r.groupby("solver"):
    reach_m = g[(g["mode"] == "multistart") & (g.category == "reachable")]
    reach_s = g[(g["mode"] == "single") & (g.category == "reachable")]
    near_m = g[(g["mode"] == "multistart") & (g.category == "near_limit")]
    bad = g[(g["mode"] == "single") & g.category.isin(["out_of_reach", "bad_orientation"])]
    rows.append(dict(solver=s,
        q1_success_single=pct(reach_s), q1_success_multistart=pct(reach_m),
        q2_median_ms=round(reach_s.time_ms.median(), 2), q2_p95_ms=round(reach_s.time_ms.quantile(.95), 2),
        q3_near_limit_multistart=pct(near_m),
        q4_false_success=int(bad.success.sum()),
        limit_violations=int((g.limit_violation_rad > 1e-6).sum())))
sc = pd.DataFrame(rows).set_index("solver")
sc.to_csv("scorecard.csv")
print("\n=== SCORECARD (reachable targets unless noted) ===")
print(sc.to_string())
print("\nQ1 = finds the answer %   Q2 = speed (ms)   Q3 = near joint limits %   Q4 = false success on impossible targets (must be 0)")

ok = sc[sc.q4_false_success == 0]
print("\n=== BEST SOLVER PER USE CASE (preliminary: simulation only, no path/real-arm data yet) ===")
print(f"Fastest (teleop / high-rate control) : {ok.q2_median_ms.idxmin()}  ({ok.q2_median_ms.min()} ms)")
def top(col):                       # all solvers tied for the best value are listed
    return " = ".join(ok.index[ok[col] == ok[col].max()]) + f"  ({ok[col].max()} %)"
print(f"Most reliable (pick & place)         : {top('q1_success_multistart')}")
print(f"Best near joint limits               : {top('q3_near_limit_multistart')}")
best = (ok.q1_success_multistart.rank(ascending=False) + ok.q2_median_ms.rank() + ok.q3_near_limit_multistart.rank(ascending=False)).idxmin()
print(f"Best all-round (rank sum of Q1,Q2,Q3): {best}")

m = r[(r["mode"] == "multistart") & (r.category == "reachable")]
fails = m[m.success == 0].groupby("target").solver.count()
all_fail = fails[fails == r.solver.nunique()].index.tolist()
print(f"\nTargets failed by ALL solvers ({len(all_fail)}): {all_fail}")
print("-> hard for every solver, so probably not a solver fault. Cause not verified yet (e.g. singular or awkward poses).")
