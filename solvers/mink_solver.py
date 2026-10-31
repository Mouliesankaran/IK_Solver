"""mink adapter (differential IK: QP step per iteration, respects joint limits).
Run alone:  python solvers/mink_solver.py"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco, mink
from common import *


def rot_from_approach(a):
    """Rotation whose +Z axis = approach vector (spin about Z is arbitrary, and is given zero weight)."""
    z = a / np.linalg.norm(a)
    x = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(x) < 1e-6: x = np.cross([0, 1.0, 0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class MinkSolver(IKSolver):
    name = "mink"
    def __init__(self, max_iter=200, dt=0.01, solver="daqp", damping=1e-3, pos_tol=1e-4, ang_tol=1e-3):
        self.max_iter, self.dt, self.solver, self.damping = max_iter, dt, solver, damping
        self.pos_tol, self.ang_tol = pos_tol, ang_tol                  # internal stop: 0.1 mm, 0.06 deg

    def setup(self, kin):
        self.kin = kin
        xml_path = kin.cfg.xml_path or str(HERE / "assets" / "so101_convert" / "so101.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        cost = np.array([1.0, 1.0, 0.0]) if kin.cfg.free_roll else np.array([1.0, 1.0, 1.0])
        self.task = mink.FrameTask(kin.cfg.ee_frame, "site", position_cost=1.0,
                                   orientation_cost=cost,
                                   lm_damping=1e-3)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def solve(self, t, q_init):
        dof = len(q_init)
        q = self.model.qpos0.copy()
        q[:dof] = q_init
        cfg = mink.Configuration(self.model, q)
        self.task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rot_from_approach(t.approach)), t.pos))
        t0 = time.perf_counter(); n = 0; ok = False
        for n in range(1, self.max_iter + 1):
            e = self.task.compute_error(cfg)
            if np.linalg.norm(e[:3]) < self.pos_tol and np.linalg.norm(e[3:5]) < self.ang_tol:
                ok = True; break
            v = mink.solve_ik(cfg, [self.task], self.dt, self.solver, damping=self.damping, limits=self.limits)
            cfg.integrate_inplace(v, self.dt)
        dt = time.perf_counter() - t0
        return IKResult(cfg.q[:dof].copy(), dt, ok, {"iters": n})


if __name__ == "__main__":
    quick_test(MinkSolver())
