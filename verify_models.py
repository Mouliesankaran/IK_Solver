"""Verification script for uploaded robot models in MuJoCo.
Usage:  python verify_models.py
"""
import mujoco
from pathlib import Path

HERE = Path(__file__).parent

MODELS = {
    "SO-101 (5-DOF)": HERE / "assets" / "so101_convert" / "scene.xml",
    "AgileX Piper (6-DOF)": HERE / "assets" / "Piper" / "scene.xml",
    "Universal Robots UR5e (6-DOF)": HERE / "assets" / "ur5e" / "scene.xml",
}

print("=" * 70)
print("MUJOCO MODEL VERIFICATION CHECK")
print("=" * 70)

for name, path in MODELS.items():
    print(f"\nChecking: {name}")
    print(f"Path:     {path}")
    if not path.exists():
        print("  ✗ File does not exist!")
        continue
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        
        joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        sites  = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        
        print("  ✅ Status: Loaded successfully!")
        print(f"  • Joints ({len(joints)}): {joints[:8]}{' ...' if len(joints) > 8 else ''}")
        print(f"  • Sites  ({len(sites)}):  {sites}")
        print(f"  • Bodies ({len(bodies)}): {len(bodies)} total")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")

print("\n" + "=" * 70)

