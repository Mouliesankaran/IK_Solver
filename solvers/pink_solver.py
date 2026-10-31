"""Pink adapter (Pinocchio-based differential IK, QP per iteration, joint limits built in).
Run alone:  python solvers/pink_solver.py"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pinocchio as pin, pink
from pink.tasks import FrameTask
from common import *
try:
    from solvers.pinocchio_solver import load_reduced_model
    from solvers.mink_solver import rot_from_approach
except ImportError:
    from pinocchio_solver import load_reduced_model
    from mink_solver import rot_from_approach


class PinkSolver(IKSolver):
    name = "pink"
    def __init__(self, max_iter=200, dt=0.01, solver="daqp", damping=1e-3, pos_tol=1e-4, ang_tol=1e-3):
        self.max_iter, self.dt, self.solver, self.damping = max_iter, dt, solver, damping
        self.pos_tol, self.ang_tol = pos_tol, ang_tol

    def setup(self, kin):
        self.kin = kin
        self.T_w_b = world_from_urdf_base(kin)
        self.model, _ = load_reduced_model(kin.cfg)
        self.data = self.model.createData()
        tip_link = "gripper_frame_link" if getattr(kin.cfg, "name", "so101") == "so101" else getattr(kin.cfg, "ee_frame", "gripper_frame")
        ori_cost = [1.0, 1.0, 0.0] if kin.cfg.free_roll else [1.0, 1.0, 1.0]
        self.task = FrameTask(tip_link, position_cost=1.0,
                              orientation_cost=ori_cost, lm_damping=1e-3)

    def solve(self, t, q_init):
        Tinv = np.linalg.inv(self.T_w_b); Rb = self.T_w_b[:3, :3]
        p = (Tinv @ np.append(t.pos, 1.0))[:3]
        Rt = Rb.T @ rot_from_approach(t.approach)
        cfg = pink.Configuration(self.model, self.data, np.clip(q_init, self.kin.lo, self.kin.hi).copy())
        self.task.set_target(pin.SE3(Rt, p))
        t0 = time.perf_counter(); ok = False; n = 0
        for n in range(1, self.max_iter + 1):
            e = self.task.compute_error(cfg)
            if np.linalg.norm(e[:3]) < self.pos_tol and np.linalg.norm(e[3:5]) < self.ang_tol:
                ok = True; break
            v = pink.solve_ik(cfg, [self.task], self.dt, solver=self.solver, damping=self.damping, safety_break=False)
            cfg.integrate_inplace(v, self.dt)
        return IKResult(cfg.q.copy(), time.perf_counter() - t0, ok, {"iters": n})


if __name__ == "__main__":
    quick_test(PinkSolver())
