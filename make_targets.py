"""Generate the FIXED target set used by every solver (seeded -> reproducible)."""
import numpy as np, csv
from common import Kin, Target, JOINTS

SEED = 42
N_REACH, N_LIMIT, N_OUT, N_BADORI = 200, 50, 40, 40
Z_SAFE = 0.015            # m above table (table top is z=0)

kin = Kin(); rng = np.random.default_rng(SEED)
lo, hi = kin.lo, kin.hi

def make(i, cat, q):
    pos, appr, _ = kin.fk(q)
    return Target(i, cat, pos, appr, q_true=q.copy(), safe_for_real=bool(pos[2] >= Z_SAFE))

targets, i = [], 0
# 1) reachable: uniform in joint limits (reachable by construction)
for _ in range(N_REACH):
    targets.append(make(i, "reachable", rng.uniform(lo, hi))); i += 1
# 2) near_limit: 1-3 joints pushed within 3-8 % of their limit
for _ in range(N_LIMIT):
    q = rng.uniform(lo, hi)
    for j in rng.choice(5, size=rng.integers(1, 4), replace=False):
        span = hi[j] - lo[j]; off = rng.uniform(0.03, 0.08) * span
        q[j] = lo[j] + off if rng.random() < 0.5 else hi[j] - off
    targets.append(make(i, "near_limit", q)); i += 1
# max reach (measured from the shoulder_lift axis)
Q = rng.uniform(lo, hi, size=(20000, 5))
reach = max(np.linalg.norm(kin.fk(q)[0] - kin.shoulder_xyz) for q in Q)
# 3) out_of_reach: push a reachable position 15-50 % beyond max reach
for _ in range(N_OUT):
    t = make(i, "out_of_reach", rng.uniform(lo, hi))
    v = t.pos - kin.shoulder_xyz; v /= np.linalg.norm(v)
    t.pos = kin.shoulder_xyz + v * reach * rng.uniform(1.15, 1.5)
    t.q_true = None; t.safe_for_real = False; targets.append(t); i += 1
# 4) bad_orientation: reachable position, approach tilted 40-90 deg sideways (out of the arm's plane)
for _ in range(N_BADORI):
    t = make(i, "bad_orientation", rng.uniform(lo, hi))
    a = t.approach; side = np.cross(a, [0, 0, 1.0]); side /= np.linalg.norm(side)
    th = np.radians(rng.uniform(40, 90)) * rng.choice([-1, 1])
    t.approach = a * np.cos(th) + side * np.sin(th)
    t.q_true = None; t.safe_for_real = False; targets.append(t); i += 1

with open("targets.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "category", "x", "y", "z", "ax", "ay", "az", "pitch_deg", "safe_for_real"] + [f"q_true_{n}" for n in JOINTS])
    for t in targets:
        w.writerow([t.id, t.category, *np.round(t.pos, 6), *np.round(t.approach, 6),
                    round(kin.pitch_deg(t.approach), 3), int(t.safe_for_real)] +
                   (list(np.round(t.q_true, 6)) if t.q_true is not None else [""] * 5))

import collections
print("max reach from shoulder_lift axis: %.3f m" % reach)
print("targets:", len(targets), dict(collections.Counter(t.category for t in targets)))
print("safe_for_real:", sum(t.safe_for_real for t in targets))
