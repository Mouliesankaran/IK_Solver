"""Common layer for the SO-101 (5-DOF) IK solver benchmark.

Every solver adapter implements IKSolver.solve(); every result is judged by the SAME
MuJoCo forward kinematics (evaluate()), so no solver can mark itself successful.

Task definition for a 5-DOF arm:
  position (x, y, z) of the `gripper_frame` site  +  approach direction (unit vector,
  site +Z axis = the direction the gripper points). Orientation about the approach axis
  (roll) is NOT constrained. Wrist roll leaves the approach axis unchanged but moves the
  tip ~8 mm off the roll axis, so the solver still needs it: 5 constraints (3 pos + 2 approach)
  for 5 joints.
"""
import time, numpy as np, mujoco
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
MODEL_XML = str(HERE / "assets" / "so101_convert" / "scene.xml")
URDF_PATH = str(HERE / "assets" / "so101_convert" / "so101_original.urdf") if (HERE / "assets" / "so101_convert" / "so101_original.urdf").exists() else str(HERE / "assets" / "so101_original.urdf")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_SITE = "gripper_frame"
HOME_Q = np.zeros(5)                      # same start seed for every solver

# success thresholds
POS_TOL_MM = 1.0
ANG_TOL_DEG = 1.0
LIMIT_TOL = 1e-6                          # rad


@dataclass
class RobotConfig:
    name: str
    dof: int
    joints: list[str]
    urdf_path: str
    xml_path: str = None
    ee_frame: str = "gripper_frame"
    free_roll: bool = False               # True for 5-DOF (roll about approach axis is unconstrained)


ROBOT_REGISTRY = {
    "so101": RobotConfig(
        name="so101",
        dof=5,
        joints=JOINTS,
        urdf_path=URDF_PATH,
        xml_path=MODEL_XML,
        ee_frame=EE_SITE,
        free_roll=True,
    ),
    "piper": RobotConfig(
        name="piper",
        dof=6,
        joints=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        urdf_path=str(HERE / "assets" / "Piper" / "piper.urdf"),
        xml_path=str(HERE / "assets" / "Piper" / "scene.xml"),
        ee_frame="gripper_frame",
        free_roll=False,
    ),
    "ur5": RobotConfig(
        name="ur5",
        dof=6,
        joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        urdf_path=str(HERE / "assets" / "ur5e" / "ur5e.urdf"),
        xml_path=str(HERE / "assets" / "ur5e" / "scene.xml"),
        ee_frame="pinch",
        free_roll=False,
    ),
    "ur5e": RobotConfig(
        name="ur5e",
        dof=6,
        joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        urdf_path=str(HERE / "assets" / "ur5e" / "ur5e.urdf"),
        xml_path=str(HERE / "assets" / "ur5e" / "scene.xml"),
        ee_frame="pinch",
        free_roll=False,
    ),
    "franka": RobotConfig(
        name="franka",
        dof=7,
        joints=[f"panda_joint{i}" for i in range(1, 8)],
        urdf_path=str(HERE / "assets" / "franka" / "panda.urdf"),
        ee_frame="panda_link8",
        free_roll=False,
    ),
}

def get_robot_config(name="so101") -> RobotConfig:
    return ROBOT_REGISTRY.get(name, ROBOT_REGISTRY["so101"])


class Kin:
    """MuJoCo-backed forward kinematics (ground truth for scoring)."""
    def __init__(self, xml=None, robot_config=None):
        if robot_config is None:
            robot_config = get_robot_config("so101")
        self.cfg = robot_config
        xml = xml or self.cfg.xml_path or MODEL_XML
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.sid = self.m.site(self.cfg.ee_frame).id
        self.jids = [self.m.joint(n).id for n in self.cfg.joints]
        self.lo = self.m.jnt_range[self.jids, 0].copy()
        self.hi = self.m.jnt_range[self.jids, 1].copy()
        mujoco.mj_kinematics(self.m, self.d)
        if "shoulder_lift" in self.cfg.joints:
            self.shoulder_xyz = self.d.xanchor[self.m.joint("shoulder_lift").id].copy()
        else:
            self.shoulder_xyz = np.zeros(3)

    def fk(self, q):
        self.d.qpos[:self.cfg.dof] = q
        mujoco.mj_kinematics(self.m, self.d)
        pos = self.d.site_xpos[self.sid].copy()
        R = self.d.site_xmat[self.sid].reshape(3, 3).copy()
        return pos, R[:, 2].copy(), R            # position, approach axis (site Z), full rotation

    def pitch_deg(self, approach):
        return np.degrees(np.arctan2(approach[2], np.hypot(approach[0], approach[1])))


@dataclass
class Target:
    id: int
    category: str                 # reachable | near_limit | out_of_reach | bad_orientation
    pos: np.ndarray
    approach: np.ndarray
    q_true: np.ndarray = None     # ground-truth joints (reachable targets only)
    safe_for_real: bool = False   # z above table + margin -> OK to try on the real arm


@dataclass
class IKResult:
    q: np.ndarray
    solve_time_s: float
    claimed_success: bool = True
    info: dict = field(default_factory=dict)


class IKSolver:
    """Adapter base class. Subclass and implement solve()."""
    name = "base"
    def setup(self, kin: Kin): ...
    def solve(self, target: Target, q_init: np.ndarray) -> IKResult:
        raise NotImplementedError


def evaluate(kin: Kin, target: Target, q):
    """Score a joint solution with the shared FK. Returns a dict of metrics."""
    q = np.asarray(q, float)
    pos, appr, _ = kin.fk(q)
    pos_err = np.linalg.norm(pos - target.pos) * 1000
    cosang = np.clip(appr @ target.approach / np.linalg.norm(target.approach), -1, 1)
    ang_err = np.degrees(np.arccos(cosang))
    viol = float(np.maximum(np.maximum(kin.lo - q, q - kin.hi), 0).max())
    return dict(pos_err_mm=pos_err, ang_err_deg=ang_err, limit_violation_rad=viol,
                success=bool(pos_err <= POS_TOL_MM and ang_err <= ANG_TOL_DEG and viol <= LIMIT_TOL))


def timed(fn, *a, **k):
    t = time.perf_counter(); out = fn(*a, **k); return out, time.perf_counter() - t


# ---------------------------------------------------------------- helpers shared by adapters
def load_targets(csv_path=None):
    """Read targets.csv back into Target objects (same order/ids for every solver)."""
    import csv
    csv_path = csv_path or str(HERE / "targets.csv")
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            q = None if r["q_true_shoulder_pan"] == "" else np.array([float(r[f"q_true_{n}"]) for n in JOINTS])
            out.append(Target(int(r["id"]), r["category"],
                              np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                              np.array([float(r["ax"]), float(r["ay"]), float(r["az"])]),
                              q, bool(int(r["safe_for_real"]))))
    return out


def world_from_urdf_base(kin):
    """4x4 transform: URDF base_link -> MuJoCo world (the two models use different root frames).
    Needed by URDF-based solvers (IKPy, Pinocchio, Placo) because targets are in MuJoCo world."""
    from urdf_fk import URDFChain
    urdf = getattr(kin.cfg, "urdf_path", URDF_PATH) if hasattr(kin, "cfg") else URDF_PATH
    tip = "gripper_frame_link" if getattr(kin.cfg, "name", "so101") == "so101" else getattr(kin.cfg, "ee_frame", "gripper_frame")
    dof = getattr(kin.cfg, "dof", 5) if hasattr(kin, "cfg") else 5
    ch = URDFChain(urdf, tip)
    q = np.zeros(dof)
    pos, _, R = kin.fk(q)
    T_w_site = np.eye(4); T_w_site[:3, :3] = R; T_w_site[:3, 3] = pos
    T = T_w_site @ np.linalg.inv(ch.fk(q))
    rng = np.random.default_rng(7)
    for _ in range(5):                               # self-check on random poses (URDF/MJCF round-off ~1e-5)
        qq = rng.uniform(kin.lo, kin.hi)
        p, _, Rr = kin.fk(qq); T2 = T @ ch.fk(qq)
        assert np.allclose(T2[:3, 3], p, atol=5e-5) and np.allclose(T2[:3, :3], Rr, atol=5e-4), "URDF/MJCF mismatch"
    return T


def quick_test(solver, n=40, seed=0):
    """Fast self-test for ONE solver:  python solvers/<name>_solver.py
    (A) sanity : start close to the true answer  -> a correct adapter must give ~100 %
    (B) real   : start from HOME_Q on reachable targets -> the honest success rate
    (C) edge   : out_of_reach / bad_orientation must NOT report success."""
    kin = Kin(); solver.setup(kin)
    tg = load_targets(); rng = np.random.default_rng(seed)
    def run(sel, init_fn, label):
        ok, ts, pe, ae = 0, [], [], []
        for t in sel:
            res = solver.solve(t, init_fn(t)); m = evaluate(kin, t, res.q)
            ok += m["success"]; ts.append(res.solve_time_s * 1000); pe.append(m["pos_err_mm"]); ae.append(m["ang_err_deg"])
        print(f"{label:<34} success {ok:>3}/{len(sel):<3} ({100*ok/len(sel):5.1f} %)  "
              f"median time {np.median(ts):7.2f} ms  median pos err {np.median(pe):8.3f} mm  median ang err {np.median(ae):7.3f} deg")
    reach = [t for t in tg if t.category == "reachable"][:n]
    run(reach, lambda t: np.clip(t.q_true + rng.normal(0, 0.02, 5), kin.lo, kin.hi), "A sanity (start near truth)")
    run(reach, lambda t: HOME_Q, "B reachable (start HOME)")
    run([t for t in tg if t.category == "near_limit"][:n], lambda t: HOME_Q, "B near_limit (start HOME)")
    for cat in ("out_of_reach", "bad_orientation"):
        run([t for t in tg if t.category == cat][:n], lambda t: HOME_Q, f"C {cat} (should be ~0 %)")
