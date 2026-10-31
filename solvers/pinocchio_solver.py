"""Pinocchio adapter: hand-written damped-least-squares IK (the classic 'CLIK' loop).
Run alone:  python solvers/pinocchio_solver.py"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pinocchio as pin
from common import *


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def load_reduced_model(robot_config=None):
    """URDF model with non-arm joints locked -> exactly the arm joints."""
    urdf = getattr(robot_config, "urdf_path", URDF_PATH) if robot_config else URDF_PATH
    full = pin.buildModelFromUrdf(urdf)
    arm_joints = robot_config.joints if robot_config else JOINTS
    lock = [full.getJointId(n) for n in full.names if n != "universe" and n not in arm_joints and full.existJointName(n)]
    if lock:
        return pin.buildReducedModel(full, lock, pin.neutral(full)), full
    return full, full


class PinocchioSolver(IKSolver):
    name = "pinocchio"
    def __init__(self, max_iter=200, damping=1e-2, step_max=0.3, pos_tol=1e-4, ang_tol=1e-3):
        self.max_iter, self.lam, self.step_max = max_iter, damping, step_max
        self.pos_tol, self.ang_tol = pos_tol, ang_tol

    def setup(self, kin):
        self.kin = kin
        self.T_w_b = world_from_urdf_base(kin)
        self.model, _ = load_reduced_model(kin.cfg)
        self.data = self.model.createData()
        tip_link = "gripper_frame_link" if getattr(kin.cfg, "name", "so101") == "so101" else getattr(kin.cfg, "ee_frame", "gripper_frame")
        self.fid = self.model.getFrameId(tip_link)
        assert self.model.nq == kin.cfg.dof, (self.model.nq, kin.cfg.dof)

    def solve(self, t, q_init):
        Tinv = np.linalg.inv(self.T_w_b)
        p = (Tinv @ np.append(t.pos, 1.0))[:3]
        a = self.T_w_b[:3, :3].T @ (t.approach / np.linalg.norm(t.approach))
        q = np.clip(np.asarray(q_init, float), self.kin.lo, self.kin.hi).copy()
        t0 = time.perf_counter(); ok = False; n = 0
        for n in range(1, self.max_iter + 1):
            pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
            M = self.data.oMf[self.fid]; a_cur = M.rotation[:, 2]
            e = np.concatenate([p - M.translation, a - a_cur])                # 3 position + 3 approach-vector rows
            if np.linalg.norm(e[:3]) < self.pos_tol and np.linalg.norm(e[3:]) < self.ang_tol:
                ok = True; break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL_WORLD_ALIGNED)
            Ja = np.vstack([J[:3], -skew(a_cur) @ J[3:]])                     # d(approach)/dq
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + self.lam**2 * np.eye(6), e)
            m = np.abs(dq).max()
            if m > self.step_max: dq *= self.step_max / m
            q = np.clip(q + dq, self.kin.lo, self.kin.hi)                     # hard joint limits
        return IKResult(q, time.perf_counter() - t0, ok, {"iters": n})


if __name__ == "__main__":
    quick_test(PinocchioSolver())
