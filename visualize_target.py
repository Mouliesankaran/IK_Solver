"""
SO-101 arm visualization for Target ID 3 — no MuJoCo needed.
Uses forward kinematics computed from URDF DH parameters (approximate).
Run:  conda activate lerobot && python visualize_target.py --target-id 3
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — saves PNG without display
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# ── Approximate FK using sequential rotation matrices ─────────────────────────
# Link lengths from SO-101 URDF (metres, approximate):
#   shoulder_lift pivot → elbow_flex pivot  ≈ 0.13 m
#   elbow_flex pivot    → wrist_flex pivot  ≈ 0.16 m
#   wrist_flex pivot    → wrist_roll pivot  ≈ 0.06 m
#   wrist_roll pivot    → EE site           ≈ 0.10 m

def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])

BASE_Z     = 0.0388   # MuJoCo world → URDF base offset in Z

def simple_fk(q):
    """
    Very approximate FK for SO-101.
    Returns list of joint pivot positions [5] + EE [1], all in world frame.
    """
    q0, q1, q2, q3, q4 = q
    p = []

    # Base pivot (shoulder_pan axis)
    p0 = np.array([0, 0, BASE_Z + 0.062])
    p.append(p0)

    # shoulder_lift pivot (shoulder_pan rotates about world Z)
    R0 = Rz(q0)
    p1 = p0 + R0 @ np.array([0, 0, 0.0])
    p.append(p1)

    # elbow_flex pivot (shoulder_lift rotates about local Y)
    R1 = R0 @ Ry(-q1)
    p2 = p1 + R1 @ np.array([0, 0, 0.13])
    p.append(p2)

    # wrist_flex pivot (elbow_flex rotates about local Y)
    R2 = R1 @ Ry(-q2)
    p3 = p2 + R2 @ np.array([0, 0, 0.16])
    p.append(p3)

    # wrist_roll pivot (wrist_flex rotates about local Y)
    R3 = R2 @ Ry(-q3)
    p4 = p3 + R3 @ np.array([0, 0, 0.06])
    p.append(p4)

    # EE (wrist_roll rotates about local Z — no position change, just orientation)
    R4 = R3 @ Rz(q4)
    p5 = p4 + R4 @ np.array([0, 0, 0.10])
    p.append(p5)

    return p, R4


def draw_arm(ax, pts, R_ee, color, label, lw=2.5, alpha=1.0):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    ax.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha, label=label)
    for p in pts[:-1]:
        ax.scatter(*p, color=color, s=50, alpha=alpha, zorder=5)
    ax.scatter(*pts[-1], color=color, s=150, marker="*", zorder=6)

    # EE approach arrow (local Z of last frame)
    ee = pts[-1]
    appr = R_ee[:, 2] * 0.06
    ax.quiver(ee[0], ee[1], ee[2],
              appr[0], appr[1], appr[2],
              color=color, linewidth=1.8, arrow_length_ratio=0.4, alpha=alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=3)
    args = parser.parse_args()

    # ── Target 3 data (from targets.csv, seed 42) ─────────────────────────
    # id=3, pos=[0.1688, 0.2363, 0.4173], approach=[0.490, 0.849, 0.196]
    # q_true = [-1.0473, 0.1905, -1.4743, 1.0865, 0.7840] rad
    TARGETS = {
        3: {
            "pos":     np.array([0.168822, 0.236285, 0.417332]),
            "approach":np.array([0.490177, 0.849277, 0.196099]),
            "q_true":  np.array([-1.047327, 0.190537, -1.474298, 1.086464, 0.784034]),
            "category":"reachable", "safe": True,
        }
    }

    if args.target_id not in TARGETS:
        print(f"Only target IDs {list(TARGETS)} are pre-loaded in this offline visualizer.")
        sys.exit(1)

    tgt = TARGETS[args.target_id]
    q_home   = np.zeros(5)
    q_target = tgt["q_true"]

    pts_home,   R_home   = simple_fk(q_home)
    pts_target, R_target = simple_fk(q_target)
    ee_home   = pts_home[-1]
    ee_target = pts_target[-1]
    cart_dist = np.linalg.norm(ee_target - ee_home) * 1000

    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    LIMITS_LO = np.degrees([-1.91986, -1.71042, -1.76278, -1.8326, -3.14159])
    LIMITS_HI = np.degrees([ 1.91986,  1.91986,  1.76278,  1.8326,  3.14159])

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0d0d1a")

    # ── 3D arm view ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(131, projection="3d")
    ax.set_facecolor("#0d0d1a")

    # Table surface
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 3), np.linspace(-0.5, 0.5, 3))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color="#aaaaaa")

    # Ground line
    ax.plot([-0.5,0.5],[0,0],[0,0], color="#555555", lw=0.8)

    draw_arm(ax, pts_home,   R_home,   color="#666666", label="Home pose",       lw=2, alpha=0.7)
    draw_arm(ax, pts_target, R_target, color="#00e5ff", label="Target solution", lw=3)

    # Target point
    ax.scatter(*tgt["pos"], color="#ff3333", s=250, marker="X", zorder=10, label="Target")
    tappr = tgt["approach"] * 0.08
    ax.quiver(*tgt["pos"], *tappr, color="#ff8888", lw=2.0, arrow_length_ratio=0.3)

    # Cartesian path
    ax.plot([ee_home[0], ee_target[0]],
            [ee_home[1], ee_target[1]],
            [ee_home[2], ee_target[2]],
            "--", color="#ffff44", lw=1.3, alpha=0.6, label=f"EE path {cart_dist:.0f} mm")

    # Base origin
    ax.scatter(0, 0, BASE_Z, color="#ffffff", s=60, zorder=5)

    ax.set_xlabel("X (m)", color="white", fontsize=8)
    ax.set_ylabel("Y (m)", color="white", fontsize=8)
    ax.set_zlabel("Z (m)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.set_title(f"SO-101  —  Target ID {args.target_id}", color="white", fontsize=10)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, loc="upper left")

    # Set equal-ish axis range
    all_p = np.array(pts_home + pts_target + [tgt["pos"]])
    mid   = all_p.mean(axis=0)
    rng   = 0.35
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(-0.05,      mid[2]+rng)
    ax.view_init(elev=20, azim=45)

    # ── Top-down view (XY) ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor("#0d0d1a")
    ax2.set_aspect("equal")

    def plot_arm_2d(ax, pts, color, label, lw=2.5, alpha=1.0):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=label)
        for p in pts[:-1]: ax.scatter(p[0], p[1], color=color, s=40, alpha=alpha)
        ax.scatter(pts[-1][0], pts[-1][1], color=color, s=120, marker="*")

    plot_arm_2d(ax2, pts_home,   "#666666", "Home", lw=2, alpha=0.7)
    plot_arm_2d(ax2, pts_target, "#00e5ff", "Target solution", lw=3)
    ax2.scatter(tgt["pos"][0], tgt["pos"][1], color="#ff3333", s=200, marker="X", zorder=10, label="Target")
    ax2.plot([ee_home[0], ee_target[0]], [ee_home[1], ee_target[1]],
             "--", color="#ffff44", lw=1.3, alpha=0.6)

    ax2.axhline(0, color="#555555", lw=0.5)
    ax2.axvline(0, color="#555555", lw=0.5)
    ax2.set_xlabel("X (m)", color="white"); ax2.set_ylabel("Y (m)", color="white")
    ax2.set_title("Top View (XY plane)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444444")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # Reach circle (max ~0.467 m)
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(0.467*np.cos(theta), 0.467*np.sin(theta),
             color="#ff6600", lw=1, alpha=0.4, linestyle=":")

    # ── Joint angle bar chart ───────────────────────────────────────────────
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor("#0d0d1a")

    y  = np.arange(5)
    ht = 0.30

    # Limit range
    for i in range(5):
        ax3.barh(y[i], LIMITS_HI[i] - LIMITS_LO[i], left=LIMITS_LO[i],
                 height=ht * 2.3, color="#2a2a4a", alpha=0.5)

    q_tgt_deg = np.degrees(q_target)
    ax3.barh(y + ht/2, np.zeros(5),      height=ht, color="#666666", label="Home (0°)")
    ax3.barh(y - ht/2, q_tgt_deg,        height=ht, color="#00e5ff", label="Target q_true")

    for i in range(5):
        ax3.text(LIMITS_LO[i]-1, y[i], f"{LIMITS_LO[i]:.0f}°", color="#ff6666",
                 va="center", ha="right", fontsize=7)
        ax3.text(LIMITS_HI[i]+1, y[i], f"{LIMITS_HI[i]:.0f}°", color="#66ff66",
                 va="center", ha="left", fontsize=7)
        ax3.text(q_tgt_deg[i] + (2 if q_tgt_deg[i] >= 0 else -2), y[i] - ht/2,
                 f"{q_tgt_deg[i]:+.1f}°", color="#00e5ff", va="center",
                 ha="left" if q_tgt_deg[i] >= 0 else "right", fontsize=8, fontweight="bold")

    ax3.set_yticks(y)
    ax3.set_yticklabels(JOINTS, color="white", fontsize=9)
    ax3.set_xlabel("Joint angle (degrees)", color="white")
    ax3.set_title("Joint Angles vs Hardware Limits", color="white")
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#444444")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax3.axvline(0, color="#888888", lw=1)

    # ── Info box under bar chart ────────────────────────────────────────────
    info = (
        f"Target ID: {args.target_id}   Category: {tgt['category']}\n"
        f"Position:  x={tgt['pos'][0]:.3f}  y={tgt['pos'][1]:.3f}  z={tgt['pos'][2]:.3f} m\n"
        f"Approach:  ({tgt['approach'][0]:.3f}, {tgt['approach'][1]:.3f}, {tgt['approach'][2]:.3f})\n"
        f"Safe for real arm: {'YES ✓' if tgt['safe'] else 'NO ✗'}\n"
        f"Cart dist home→target: {cart_dist:.1f} mm\n"
        f"Joint dist home→target: {np.sum(np.abs(q_tgt_deg)):.1f}°"
    )
    ax3.text(0.01, -0.28, info, transform=ax3.transAxes, color="#cccccc",
             fontsize=8, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.85))

    plt.suptitle(f"SO-101  ·  Target ID {args.target_id}  ·  Benchmark Point",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()

    outpath = HERE / f"target_{args.target_id}_pose.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"Saved → {outpath}")

if __name__ == "__main__":
    main()
