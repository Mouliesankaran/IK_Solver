"""SO-101 Real-Robot IK Benchmark — simplified interactive version
==================================================================
Usage:
  python real_robot_benchmark.py --target-id 3
  python real_robot_benchmark.py          (manual coordinates)

What it measures per solver:
  1. IK solve time (ms)
  2. Cartesian distance — home EE → target (mm)
  3. Joint space distance — sum |Δq| in degrees
  4. Torque proxy — mean Present_Load % while moving
  5. Approx power (mW)

  + Joint states: commanded vs actual after settling

Flow: connect → pick solver (1-4) → confirm → measure → table → repeat
Ctrl+C at any time → torque off, safe exit.
"""

import argparse, csv, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from common import Kin, load_targets, evaluate, HOME_Q, JOINTS, EE_SITE, Target
from report_generator import generate_individual_report, generate_comparison_report

# ── LeRobot ──────────────────────────────────────────────────────────────────
try:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
except ImportError as e:
    print(f"[ERROR] LeRobot not found: {e}\nRun:  conda activate lerobot")
    sys.exit(1)

# ── Solver registry ───────────────────────────────────────────────────────────
SOLVER_MAP = {}
for name, cls_name, mod in [
    ("ikpy",      "IKPySolver",      "solvers.ikpy_solver"),
    ("mink",      "MinkSolver",      "solvers.mink_solver"),
    ("pink",      "PinkSolver",      "solvers.pink_solver"),
    ("pinocchio", "PinocchioSolver", "solvers.pinocchio_solver"),
]:
    try:
        import importlib
        mod_obj = importlib.import_module(mod)
        SOLVER_MAP[name] = getattr(mod_obj, cls_name)
    except Exception:
        pass

# ── Constants & Directories ──────────────────────────────────────────────────
ARM_JOINTS     = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SLOW_SPEED     = 100          # STS3215 speed register
SETTLE_TOL     = 0.5          # deg — arm is "settled" when all joints within this range
SETTLE_WINDOW  = 5            # consecutive stable reads needed
SETTLE_TIMEOUT = 12.0         # max seconds to wait for settling
LOAD_SAMPLE_HZ = 20           # Hz — how fast we poll Present_Load while moving
VOLTAGE_SCALE  = 0.1          # register × 0.1 = Volts
RATED_CURRENT  = 3200.0       # mA at 100% load (STS3215 stall approx)

# Clean single directory for ALL outputs (no loose datasets or reports)
OUTPUT_DIR = HERE / "real_benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CSV = OUTPUT_DIR / "real_results.csv"
SOLVER_REPORTS_DIR = OUTPUT_DIR / "solver_reports"
SOLVER_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_REPORT_FILE = OUTPUT_DIR / "comparison_report.md"

RESULT_FIELDS = [
    "timestamp", "target_id", "pos_x", "pos_y", "pos_z", "approach_x", "approach_y", "approach_z",
    "solver",
    "ik_solve_ms",
    "cartesian_dist_mm",
    "joint_dist_deg",
    "mean_torque_pct", "approx_power_mW",
    "settle_time_ms",
    "sim_pos_err_mm",
    "real_pos_err_mm",
    "cmd_shoulder_pan", "cmd_shoulder_lift", "cmd_elbow_flex",
    "cmd_wrist_flex", "cmd_wrist_roll",
    "actual_shoulder_pan", "actual_shoulder_lift", "actual_elbow_flex",
    "actual_wrist_flex", "actual_wrist_roll",
    "notes",
]

# ─────────────────────────── unit helpers ────────────────────────────────────

def d2r(deg):  return np.asarray(deg) * np.pi / 180.0
def r2d(rad):  return np.asarray(rad) * 180.0 / np.pi

def obs_to_rad(obs: dict) -> np.ndarray:
    return d2r([obs[f"{j}.pos"] for j in ARM_JOINTS])

def rad_to_action(q_rad: np.ndarray) -> dict:
    q_deg = r2d(q_rad)
    act   = {f"{j}.pos": float(q_deg[i]) for i, j in enumerate(ARM_JOINTS)}
    act["gripper.pos"] = 0.0
    return act

# ─────────────────────────── motor helpers ───────────────────────────────────

def set_goal_speed(bus, speed: int):
    for j in ARM_JOINTS:
        try:
            bus.write("Moving_Speed", j, speed)
        except Exception:
            pass

def read_load_pct(bus) -> float:
    """Return mean Present_Load (0–100%) across all 5 arm joints."""
    try:
        raw = bus.sync_read("Present_Load", normalize=False)
        vals = [(raw[j] & 0x3FF) for j in ARM_JOINTS if j in raw]
        return np.mean(vals) * (100.0 / 1000.0) if vals else 0.0
    except Exception:
        return 0.0

def read_voltage(bus) -> float:
    try:
        raw = bus.sync_read("Present_Voltage", motors="shoulder_pan", normalize=False)
        return raw["shoulder_pan"] * VOLTAGE_SCALE
    except Exception:
        return 12.0   # safe default

def wait_settled(robot) -> float:
    """Poll until joints stop moving. Returns settle_time_ms."""
    history = []
    t0 = time.perf_counter()
    while True:
        obs = robot.get_observation()
        pos = [obs[f"{j}.pos"] for j in ARM_JOINTS]
        history.append(pos)
        elapsed = time.perf_counter() - t0
        if elapsed > SETTLE_TIMEOUT:
            break
        if len(history) >= SETTLE_WINDOW:
            recent = np.array(history[-SETTLE_WINDOW:])
            if (recent.max(axis=0) - recent.min(axis=0)).max() < SETTLE_TOL:
                break
        time.sleep(1.0 / LOAD_SAMPLE_HZ)
    return (time.perf_counter() - t0) * 1000

def move_smooth(robot: SOFollower, q_target_deg: np.ndarray, speed_deg_s: float = 25.0, rate_hz: int = 30) -> tuple[float, float, float]:
    """
    Interpolate smoothly between current joints and q_target_deg using a cosine S-curve trajectory.
    This prevents violent jumps, gear stress, and high motor overshoot.
    Samples Present_Load during the actual trajectory.
    Returns: (mean_load_pct, power_mW, move_duration_s)
    """
    obs = robot.get_observation()
    q_start = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS])
    q_target = np.asarray(q_target_deg, dtype=float)

    max_delta = float(np.max(np.abs(q_target - q_start)))
    duration_s = max(max_delta / max(speed_deg_s, 1.0), 0.6)
    steps = max(int(duration_s * rate_hz), 8)
    dt = duration_s / steps

    loads = []
    voltage = read_voltage(robot.bus)
    t_start = time.perf_counter()

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        # Cosine S-curve: smooth start and smooth stop (zero jerk at boundaries)
        frac = step / steps
        s = 0.5 * (1.0 - np.cos(np.pi * frac))
        q_interp = q_start + s * (q_target - q_start)

        act = {f"{j}.pos": float(q_interp[i]) for i, j in enumerate(ARM_JOINTS)}
        act["gripper.pos"] = 0.0
        robot.send_action(act)

        # Sample load during motion
        if step % 2 == 0:
            loads.append(read_load_pct(robot.bus))

        elapsed_step = time.perf_counter() - t0
        sleep_time = dt - elapsed_step
        if sleep_time > 0:
            time.sleep(sleep_time)

    actual_duration = time.perf_counter() - t_start
    mean_load = float(np.mean(loads)) if loads else 0.0
    power_mW = voltage * (mean_load / 100.0) * RATED_CURRENT
    return mean_load, power_mW, actual_duration

# ─────────────────────────── CSV writer ──────────────────────────────────────

def save_row(row: dict):
    exists = RESULT_CSV.exists()
    with open(RESULT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)

# ─────────────────────────── pretty table ────────────────────────────────────

def print_result_table(results: list[dict]):
    """Print a clean comparison table after all solvers finish."""
    if not results:
        return
    cols = [
        ("Solver",        "solver",           "<10"),
        ("IK (ms)",       "ik_solve_ms",      ">8.1f"),
        ("Cart dist(mm)", "cartesian_dist_mm",">13.1f"),
        ("Joint dist(°)", "joint_dist_deg",   ">13.1f"),
        ("Torque %",      "mean_torque_pct",  ">9.1f"),
        ("Power (mW)",    "approx_power_mW",  ">10.0f"),
        ("Settle(ms)",    "settle_time_ms",   ">10.0f"),
        ("Sim err(mm)",   "sim_pos_err_mm",   ">11.2f"),
        ("Real err(mm)",  "real_pos_err_mm",  ">12.2f"),
    ]
    header = "  ".join(f"{h:{spec.split('.')[0].replace('>','').replace('<','').replace('^','')}}"
                       for h, _, spec in cols)
    print("\n" + "=" * 100)
    print("  REAL-ROBOT BENCHMARK RESULTS")
    print("=" * 100)
    # Simple aligned header
    widths = [13, 8, 13, 13, 9, 10, 10, 11, 12]
    hdr = ("  ".join(h.ljust(w) for (h, _, _), w in zip(cols, widths)))
    print(hdr)
    print("-" * 100)
    for r in results:
        row_str = []
        for (_, key, spec), w in zip(cols, widths):
            val = r.get(key, "")
            if isinstance(val, float):
                row_str.append(f"{val:{spec}}".ljust(w))
            else:
                row_str.append(str(val).ljust(w))
        print("  ".join(row_str))
    print("=" * 100)

    # Joint states per solver
    print("\n  JOINT STATES (commanded vs actual, degrees):")
    print(f"  {'Solver':<12} {'Joint':<15} {'Commanded':>10} {'Actual':>10} {'Error':>8}")
    print("  " + "-" * 55)
    for r in results:
        for j in ARM_JOINTS:
            cmd = r.get(f"cmd_{j}", 0.0)
            act = r.get(f"actual_{j}", 0.0)
            err = act - cmd
            print(f"  {r['solver']:<12} {j:<15} {cmd:>10.2f} {act:>10.2f} {err:>+8.2f}")
        print("  " + "-" * 55)

# ─────────────────────────── safety ──────────────────────────────────────────

def safety_ok(kin: Kin, q: np.ndarray, target: Target) -> bool:
    m = evaluate(kin, target, q)
    if m["limit_violation_rad"] > 1e-6:
        print(f"  ✗ JOINT LIMIT violated ({m['limit_violation_rad']:.4f} rad) — skip")
        return False
    if m["pos_err_mm"] > 10.0:
        print(f"  ✗ Sim error too large ({m['pos_err_mm']:.1f} mm > 10 mm) — skip")
        return False
    return True

# ─────────────────────────── one solver run ──────────────────────────────────

def run_one_solver(name: str, target: Target, kin: Kin, robot: SOFollower,
                   home_ee_pos: np.ndarray, speed_deg_s: float = 25.0) -> dict | None:
    print(f"\n  ┌── Solver: {name.upper()} ─────────────────────────────────────────")

    # Instantiate & setup
    try:
        solver = SOLVER_MAP[name]()
        solver.setup(kin)
    except Exception as e:
        print(f"  ✗ Setup failed: {e}"); return None

    # IK solve
    res = solver.solve(target, HOME_Q)
    ik_ms = res.solve_time_s * 1000
    q_sol = res.q
    print(f"  IK solved in {ik_ms:.1f} ms  |  claimed={res.claimed_success}")

    # Sim evaluation
    m_sim = evaluate(kin, target, q_sol)
    print(f"  Sim pos_err = {m_sim['pos_err_mm']:.2f} mm  |  ang_err = {m_sim['ang_err_deg']:.2f}°")

    # Safety check
    if not safety_ok(kin, q_sol, target):
        return None

    # Distances
    cartesian_dist = np.linalg.norm(target.pos - home_ee_pos) * 1000   # mm
    q_now = obs_to_rad(robot.get_observation())
    joint_dist = float(np.sum(np.abs(r2d(q_sol - q_now))))             # deg

    # Show commanded joints
    q_deg = r2d(q_sol)
    print(f"\n  Commanded joints (deg):")
    for j, v in zip(ARM_JOINTS, q_deg):
        print(f"    {j:<18} = {v:+.2f}°")
    print(f"\n  Cartesian distance (home→target): {cartesian_dist:.1f} mm")
    print(f"  Joint space distance:             {joint_dist:.1f}°")

    # Confirm
    ans = input("\n  Press ENTER to send to arm, or 's' to skip: ").strip().lower()
    if ans == "s":
        print("  Skipped."); return None

    # Smooth trajectory execution
    print(f"  Moving smoothly (speed: {speed_deg_s:.1f}°/s)...")
    mean_load, power_mW, actual_dur = move_smooth(robot, q_deg, speed_deg_s=speed_deg_s)
    print(f"  Move completed in {actual_dur:.1f}s | Torque: {mean_load:.1f}% | Power≈{power_mW:.0f} mW")

    # Wait for settling
    settle_ms = wait_settled(robot)
    print(f"  Settled in {settle_ms:.0f} ms")

    # Read actual joints
    obs = robot.get_observation()
    actual_q_deg = [obs[f"{j}.pos"] for j in ARM_JOINTS]
    actual_q_rad = d2r(actual_q_deg)
    cmd_q_deg    = list(q_deg)

    # Real arm evaluation
    m_real = evaluate(kin, target, actual_q_rad)
    print(f"  Real pos_err = {m_real['pos_err_mm']:.2f} mm  |  ang_err = {m_real['ang_err_deg']:.2f}°")
    print(f"  └─────────────────────────────────────────────────────────────")

    row = {
        "timestamp": datetime.now().isoformat(),
        "target_id": getattr(target, "id", -1),
        "pos_x": round(float(target.pos[0]), 4),
        "pos_y": round(float(target.pos[1]), 4),
        "pos_z": round(float(target.pos[2]), 4),
        "approach_x": round(float(target.approach[0]), 3),
        "approach_y": round(float(target.approach[1]), 3),
        "approach_z": round(float(target.approach[2]), 3),
        "solver": name,
        "ik_solve_ms": round(ik_ms, 2),
        "cartesian_dist_mm": round(cartesian_dist, 1),
        "joint_dist_deg": round(joint_dist, 1),
        "mean_torque_pct": round(mean_load, 2),
        "approx_power_mW": round(power_mW, 1),
        "settle_time_ms": round(settle_ms, 1),
        "sim_pos_err_mm": round(m_sim["pos_err_mm"], 3),
        "real_pos_err_mm": round(m_real["pos_err_mm"], 3),
        **{f"cmd_{j}":    round(cmd_q_deg[i], 3)    for i, j in enumerate(ARM_JOINTS)},
        **{f"actual_{j}": round(actual_q_deg[i], 3) for i, j in enumerate(ARM_JOINTS)},
        "notes": "",
    }
    save_row(row)
    indiv_rep = generate_individual_report(row, SOLVER_REPORTS_DIR)
    print(f"  [Report] Individual report saved → {indiv_rep.relative_to(HERE)}")
    return row

# ─────────────────────────── return home ─────────────────────────────────────

def go_home(robot: SOFollower, speed_deg_s: float = 25.0):
    print(f"\n  Returning to home smoothly (all zeros, {speed_deg_s:.1f}°/s)...")
    move_smooth(robot, np.zeros(5), speed_deg_s=speed_deg_s)
    wait_settled(robot)
    time.sleep(0.3)

# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--port",      default="/dev/ttyACM0")
    parser.add_argument("--robot-id",  default="my_awesome_follower_arm")
    parser.add_argument("--speed",     type=float, default=20.0,
                        help="Joint speed in degrees/second. Default=20.0 (smooth & safe)")
    args = parser.parse_args()

    # ── Load MuJoCo ──────────────────────────────────────────────────────
    print("\nLoading MuJoCo model...")
    kin = Kin()

    # ── Get target ────────────────────────────────────────────────────────
    all_targets = load_targets()

    if args.target_id is not None:
        tgt = next((t for t in all_targets if t.id == args.target_id), None)
        if tgt is None:
            print(f"Target ID {args.target_id} not found."); sys.exit(1)
        if not tgt.safe_for_real:
            print(f"[WARNING] Target {args.target_id} is NOT safe_for_real (z < 0.015 m)")
            if input("Continue anyway? (y/N): ").strip().lower() != "y":
                sys.exit(0)
    else:
        print("\n" + "=" * 65)
        print("TARGET COORDINATES (MuJoCo World Frame)")
        print("=" * 65)
        print("Verified Safe Benchmark Point (Target ID 3):")
        print("  X:  0.1688 m  (forward reach)")
        print("  Y:  0.2363 m  (left lateral)")
        print("  Z:  0.4173 m  (height above table)")
        print("  Approach Vector: ax=0.490, ay=0.849, az=0.196")
        print("-" * 65)
        choice = input("Press ENTER to use this point, or type 'c' to enter custom: ").strip().lower()
        if choice != "c":
            x, y, z = 0.168822, 0.236285, 0.417332
            ax, ay, az = 0.490177, 0.849277, 0.196099
            tid = 3
        else:
            x  = float(input("  x (m): "))
            y  = float(input("  y (m): "))
            z  = float(input("  z (m): "))
            ax = float(input("  approach x: "))
            ay = float(input("  approach y: "))
            az = float(input("  approach z: "))
            tid = -1
        appr = np.array([ax, ay, az]); appr /= np.linalg.norm(appr)
        tgt  = Target(id=tid, category="manual", pos=np.array([x,y,z]),
                      approach=appr, q_true=None, safe_for_real=(z >= 0.015))

    pos_home, _, _ = kin.fk(HOME_Q)
    print(f"\nTarget: pos={np.round(tgt.pos,4)}  approach={np.round(tgt.approach,3)}")
    print(f"Cartesian dist home→target: {np.linalg.norm(tgt.pos-pos_home)*1000:.1f} mm")

    # ── Connect robot ────────────────────────────────────────────────────
    print(f"\nConnecting to SO-101 on {args.port}...")
    cfg    = SOFollowerRobotConfig(port=args.port, id=args.robot_id, use_degrees=True)
    robot  = SOFollower(cfg)

    available = list(SOLVER_MAP.keys())
    if not available:
        print("[ERROR] No solvers loaded."); sys.exit(1)

    all_results = []

    try:
        robot.connect(calibrate=False)
        print(f"Connected. Movement speed = {args.speed:.1f}°/s")

        print(f"\nAvailable solvers: {available}")

        while True:
            print("\n" + "─" * 50)
            print("Choose solver to test:")
            for i, name in enumerate(available, 1):
                print(f"  {i}. {name}")
            print("  q. Quit and generate reports")
            ans = input("Choice: ").strip().lower()

            if ans == "q":
                break
            try:
                idx  = int(ans) - 1
                name = available[idx]
            except (ValueError, IndexError):
                print("  Invalid choice."); continue

            row = run_one_solver(name, tgt, kin, robot, pos_home, speed_deg_s=args.speed)
            if row:
                all_results.append(row)
                print_result_table(all_results)

            if input("\nTest another solver? (ENTER=yes, q=quit): ").strip().lower() == "q":
                break

            go_home(robot, speed_deg_s=args.speed)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] Disabling torque and exiting safely.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        try:
            robot.disconnect()
            print("Robot disconnected.")
        except Exception:
            pass

    print_result_table(all_results)
    if all_results:
        comp_rep = generate_comparison_report(all_results, COMPARISON_REPORT_FILE)
        print("\n" + "=" * 65)
        print("ALL BENCHMARK OUTPUTS SAVED CLEANLY IN ONE FOLDER:")
        print(f"  Folder:             {OUTPUT_DIR.relative_to(HERE)}/")
        print(f"  Raw Dataset:        {RESULT_CSV.relative_to(HERE)}")
        print(f"  Individual Reports: {SOLVER_REPORTS_DIR.relative_to(HERE)}/<solver>_report.md")
        print(f"  Comparison Report:  {comp_rep.relative_to(HERE)}")
        print("=" * 65)


if __name__ == "__main__":
    main()
