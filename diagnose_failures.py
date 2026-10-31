"""Failure Diagnostic for Hard Targets: python diagnose_failures.py

Analyzes the 6 reachable targets that failed across all solvers (5, 59, 105, 134, 148, 182):
1. Workspace position & reach radius relative to shoulder pivot
2. Proximity of ground-truth joints (q_true) to limits
3. Jacobian conditioning (singularities & manipulability at q_true)
4. Deep multistart test (50 random restarts vs baseline 5)
"""
import numpy as np
import mujoco
from pathlib import Path
from common import Kin, load_targets, JOINTS, EE_SITE, evaluate, HOME_Q

HERE = Path(__file__).parent
HARD_IDS = [5, 59, 105, 134, 148, 182]


def compute_task_jacobian(kin: Kin, q: np.ndarray):
    """5x5 task Jacobian at q: 3 position + 2 approach vector constraints."""
    kin.d.qpos[:5] = q
    mujoco.mj_kinematics(kin.m, kin.d)
    mujoco.mj_comPos(kin.m, kin.d)

    jacp = np.zeros((3, kin.m.nv))
    jacr = np.zeros((3, kin.m.nv))
    mujoco.mj_jacSite(kin.m, kin.d, jacp, jacr, kin.sid)

    J_pos = jacp[:, :5]      # (3, 5)
    J_rot = jacr[:, :5]      # (3, 5)

    pos, appr, R = kin.fk(q) # appr is (3,) unit vector
    # Derivative of approach vector: da/dt = omega x a = -[a]_x * omega
    skew_a = np.array([
        [0, -appr[2], appr[1]],
        [appr[2], 0, -appr[0]],
        [-appr[1], appr[0], 0]
    ])
    J_da = -skew_a @ J_rot   # (3, 5)

    # Project onto 2D subspace orthogonal to approach vector
    # Using the other two columns of the site orientation frame R = [x, y, z]
    R_x = R[:, 0]
    R_y = R[:, 1]
    J_appr_2d = np.vstack([R_x @ J_da, R_y @ J_da])  # (2, 5)

    J_task = np.vstack([J_pos, J_appr_2d])           # (5, 5)
    return J_task, J_pos


def analyze_target(kin: Kin, target):
    """Extract physical and kinematic metrics for a target."""
    q = target.q_true
    shoulder = kin.shoulder_xyz
    reach_dist = np.linalg.norm(target.pos - shoulder)
    xy_dist = np.hypot(target.pos[0] - shoulder[0], target.pos[1] - shoulder[1])

    # Margin to joint limits in % of full range
    ranges = kin.hi - kin.lo
    dist_lo = (q - kin.lo) / ranges * 100
    dist_hi = (kin.hi - q) / ranges * 100
    margin_pct = np.minimum(dist_lo, dist_hi)
    min_margin_joint = JOINTS[np.argmin(margin_pct)]
    min_margin_val = np.min(margin_pct)

    # Jacobian conditioning
    J_task, J_pos = compute_task_jacobian(kin, q)
    s = np.linalg.svd(J_task, compute_uv=False)
    cond_num = s[0] / max(s[-1], 1e-12)
    manipulability = np.prod(s)  # sqrt(det(J J^T)) for square matrix is abs(det(J)) = product of SVs

    return {
        "id": target.id,
        "reach_mm": reach_dist * 1000,
        "xy_mm": xy_dist * 1000,
        "z_mm": target.pos[2] * 1000,
        "pitch_deg": target.pitch_deg(target.approach),
        "min_margin_joint": min_margin_joint,
        "min_margin_pct": min_margin_val,
        "min_sv": s[-1],
        "cond_num": cond_num,
        "manip": manipulability,
        "q_true": np.round(q, 3).tolist()
    }


def test_deep_multistart(kin: Kin, target, max_restarts=50):
    """Test if pink solver converges when given more random restarts."""
    from solvers.pink_solver import PinkSolver
    solver = PinkSolver()
    solver.setup(kin)

    # 1. Start from HOME
    res = solver.solve(target, HOME_Q)
    if evaluate(kin, target, res.q)["success"]:
        return 1, True

    # 2. Random restarts
    rng = np.random.default_rng(2000 + target.id)
    for attempt in range(2, max_restarts + 1):
        q_rand = rng.uniform(kin.lo, kin.hi)
        res = solver.solve(target, q_rand)
        if evaluate(kin, target, res.q)["success"]:
            return attempt, True

    return max_restarts, False


def main():
    kin = Kin()
    targets = {t.id: t for t in load_targets() if t.id in HARD_IDS}

    print("=" * 80)
    print("DIAGNOSTIC: 6 HARD TARGETS FAILED BY ALL SOLVERS")
    print("=" * 80)
    print(f"{'ID':>4} | {'Reach(mm)':>9} | {'Pitch(°)':>8} | {'Tight Joint':>13} | {'Limit Margin':>12} | {'Task Cond #':>11} | {'Manip':>10}")
    print("-" * 80)

    stats = []
    for tid in HARD_IDS:
        t = targets[tid]
        st = analyze_target(kin, t)
        stats.append(st)
        print(f"{st['id']:>4} | {st['reach_mm']:>9.1f} | {st['pitch_deg']:>8.1f} | "
              f"{st['min_margin_joint']:>13} | {st['min_margin_pct']:>11.1f}% | "
              f"{st['cond_num']:>11.1f} | {st['manip']:>10.2e}")

    print("\n" + "=" * 80)
    print("DEEP MULTISTART TEST (Pink solver with up to 50 restarts)")
    print("=" * 80)
    for tid in HARD_IDS:
        t = targets[tid]
        attempts, ok = test_deep_multistart(kin, t, max_restarts=50)
        status = f"SOLVED on attempt #{attempts}" if ok else f"FAILED after {attempts} restarts"
        print(f"Target {tid:>3}: {status}")

    print("\n" + "=" * 80)
    print("ROOT-CAUSE SUMMARY")
    print("=" * 80)
    for st in stats:
        reasons = []
        if st['min_margin_pct'] < 5.0:
            reasons.append(f"Joint '{st['min_margin_joint']}' is at {st['min_margin_pct']:.1f}% from hardware limit (boundary lock)")
        if st['cond_num'] > 100.0:
            reasons.append(f"Near-singular task Jacobian (condition number = {st['cond_num']:.1f})")
        if st['reach_mm'] > 440:
            reasons.append(f"Near maximum arm reach boundary ({st['reach_mm']:.1f} mm / 467 mm)")
        if not reasons:
            reasons.append("Complex non-linear coupling / narrow attraction basin")
        print(f"Target {st['id']}: {', '.join(reasons)}")


if __name__ == "__main__":
    main()

