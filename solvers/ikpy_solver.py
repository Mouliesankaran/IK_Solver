"""IKPy adapter (numerical, scipy least-squares).   Run alone:  python solvers/ikpy_solver.py"""
import sys, warnings; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial.transform import Rotation as R
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from common import *
from urdf_fk import URDFChain


class IKPySolver(IKSolver):
    name = "ikpy"
    def __init__(self, optimizer="least_squares", budget=None):
        self.optimizer, self.budget = optimizer, budget

    def setup(self, kin):
        self.kin = kin
        self.T_w_b = world_from_urdf_base(kin)                 # world <- URDF base
        urdf_path = getattr(kin.cfg, "urdf_path", URDF_PATH)
        tip_link = "gripper_frame_link" if getattr(kin.cfg, "name", "so101") == "so101" else getattr(kin.cfg, "ee_frame", "gripper_frame")
        urdf = URDFChain(urdf_path, tip_link)
        links, k = [OriginLink()], 0
        self.active_mask = [False]
        for j in urdf.chain:
            T = j["T"]; xyz = T[:3, 3]; rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
            if j["type"] == "revolute":
                links.append(URDFLink(j["name"], xyz, rpy, rotation=j["axis"], bounds=(kin.lo[k], kin.hi[k])))
                self.active_mask.append(True)
                k += 1
            else:
                links.append(URDFLink(j["name"], xyz, rpy, joint_type="fixed"))
                self.active_mask.append(False)
        self.chain = Chain(links, active_links_mask=self.active_mask, name=kin.cfg.name)

    def solve(self, t, q_init):
        Rb = self.T_w_b[:3, :3]
        p = np.linalg.inv(self.T_w_b) @ np.append(t.pos, 1.0)    # target -> URDF base frame
        a = Rb.T @ (t.approach / np.linalg.norm(t.approach))
        x0 = np.zeros(len(self.active_mask))
        rev_idx = [i for i, act in enumerate(self.active_mask) if act]
        x0[rev_idx] = np.clip(q_init, self.kin.lo, self.kin.hi)
        kw = dict(optimizer=self.optimizer)
        if self.budget: kw["optimizer_budget"] = self.budget
        t0 = time.perf_counter()
        sol = self.chain.inverse_kinematics(p[:3], a, orientation_mode="Z", initial_position=x0, **kw)
        dt = time.perf_counter() - t0
        q_sol = np.array([sol[i] for i in rev_idx])
        return IKResult(q_sol, dt)


if __name__ == "__main__":
    quick_test(IKPySolver())
