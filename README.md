# Multi-Robot Inverse Kinematics (IK) Benchmarking & Real-Time Hardware Suite

A comprehensive robotics engineering framework for benchmarking, analyzing, and deploying multiple Inverse Kinematics (IK) solvers across real physical hardware and high-fidelity MuJoCo simulations.

Designed for low-latency robotics control, visual servoing, and autonomous manipulation.

---

## 1. Supported Robots & Kinematics

| Robot Arm | Degrees of Freedom (DOF) | Kinematic Formulation | Native Simulation Model | URDF Kinematic Model | End-Effector Tracking Frame |
|---|---|---|---|---|---|
| **SO-101 Follower Arm** | **5-DOF** (Pitch-Roll Wrist) | 3D Position + 2D Approach Axis (Free Roll about Z) | `assets/so101_convert/scene.xml` | `assets/so101_convert/so101_original.urdf` | `gripper_frame` |
| **AgileX Piper** | **6-DOF** (Spherical Wrist) | Full 6D SE(3) Pose (Position + Orientation) | `assets/Piper/scene.xml` | `assets/Piper/piper_description.urdf` | `gripper_frame` |
| **Universal Robots UR5e** | **6-DOF** (Industrial Arm) | Full 6D SE(3) Pose (Position + Orientation) | `assets/ur5e/scene.xml` | `assets/ur5e/urdf/ur5e.urdf` | `pinch` (Fingertip center) |

---

## 2. Integrated IK Solvers

1. **Pinocchio (`pinocchio`)**
   - **Type:** Closed-Loop Inverse Kinematics (CLIK) via Damped Least-Squares Jacobian.
   - **Performance:** **Fastest solver** (~0.35 – 0.85 ms solve latency).
   - **Best For:** Ultra-high-frequency teleoperation loops (>500 Hz) and real-time visual tracking.

2. **Mink (`mink`)**
   - **Type:** Quadratic Programming (QP)-based differential IK formulated on MuJoCo MJCF with hard joint limits.
   - **Performance:** Robust convergence (~1.8 – 30 ms).
   - **Best For:** Native MuJoCo simulation, contact-rich tasks, and strict avoidance of joint limit violations.

3. **Pink (`pink`)**
   - **Type:** QP-based differential IK built on Pinocchio with configurable Cartesian task weights.
   - **Performance:** Clean velocity-bounded convergence (~15 – 25 ms).
   - **Best For:** Smooth motion planning with joint velocity/acceleration constraints.

4. **IKPy (`ikpy`)**
   - **Type:** Numerical optimization (SciPy Levenberg-Marquardt / Least Squares).
   - **Performance:** High Cartesian precision (0.000 mm position error), moderate latency (~8 – 28 ms).
   - **Best For:** Offline trajectory generation, waypoint validation, and sub-millimeter precision placement.

---

## 3. Real Robot vs. Simulation Parity

- **Trajectory Smoothing:** Both hardware ([real_robot_benchmark.py](file:///home/moulie/Downloads/IK_Solver/real_robot_benchmark.py)) and simulation ([sim_live_viewer.py](file:///home/moulie/Downloads/IK_Solver/sim_live_viewer.py)) use an identical **cosine S-curve trajectory generator** (`smooth_interpolate` at 20°/s) to prevent abrupt motor torque spikes and physical arm snapping.
- **Unified UX & Telemetry:** Both runners present the same interactive menus, report progress, and generate structured Markdown and CSV scorecards.
- **Output Discipline:** All generated data is kept cleanly isolated in dedicated folders (`real_benchmark_results/` and `sim_benchmark_results/`).

---

## 4. Environment & Installation

Activate the Conda environment containing MuJoCo, Pinocchio, and LeRobot dependencies:

```bash
conda activate lerobot
```

Ensure the required Python packages are installed:

```bash
pip install mujoco mink pink pinocchio ikpy scipy numpy tabulate
```

For real hardware control (Feetech STS3215 bus servos on SO-101):
```bash
pip install feetech-servo-sdk
```

---

## 5. Usage & CLI Command Guide

### A. Real Robot Interactive Benchmark (SO-101 Hardware)
Connect the real SO-101 arm via USB serial bus adapter:

```bash
# 1. Interactive target selection & solver benchmarking
python real_robot_benchmark.py

# 2. Benchmark on a specific target ID
python real_robot_benchmark.py --target-id 3

# 3. Specify custom serial port and speed (deg/s)
python real_robot_benchmark.py --port /dev/ttyUSB0 --speed 20.0
```

### B. 3D Live Simulation Viewer & Interactive Benchmark (MuJoCo)
Visualize the robot live in a GLFW 3D window, test solvers, inspect joint movements, and generate comparison reports:

```bash
# 1. SO-101 Arm (Default)
python sim_live_viewer.py

# 2. AgileX Piper 6-DOF
python sim_live_viewer.py --robot piper

# 3. Universal Robots UR5e 6-DOF
python sim_live_viewer.py --robot ur5e

# 4. Direct solver execution (headless/scriptable without menu prompt)
python sim_live_viewer.py --robot ur5e --solver pinocchio
python sim_live_viewer.py --robot piper --solver mink
python sim_live_viewer.py --robot so101 --solver ikpy

# 5. Run ALL solvers sequentially on a target in simulation
python sim_live_viewer.py --robot ur5e --solver all
```

### C. Batch Benchmark Suite (All Workspace Targets)
Evaluate global success rates, near-limit robustness, and algorithmic latency across targets in `targets.csv`:

```bash
# 1. Run all solvers on SO-101
python benchmark.py --robot so101

# 2. Benchmark specific solvers on UR5e or Piper
python benchmark.py --robot ur5e --solvers pinocchio mink ikpy
python benchmark.py --robot piper --solvers pinocchio mink

# 3. Quick sanity check on the first 10 targets
python benchmark.py --robot so101 --limit 10
```

### D. Diagnostics & Model Verification
```bash
# Verify that all MuJoCo models (SO-101, Piper, UR5e) compile cleanly
python verify_models.py

# Analyze hard-to-reach targets, singularities, and manipulability
python diagnose_failures.py

# Re-generate reachable target points dataset
python make_targets.py

# Headless visualizer for target pose snapshot
python visualize_target.py --target-id 3
```

---

## 6. Annotated Project File Structure

Every file and directory in this workspace serves a specific, documented purpose:

```text
IK_Solver/
│
├── assets/                                      # Kinematic model definitions, meshes, and scenes
│   ├── Piper/                                   # AgileX Piper 6-DOF robot assets
│   │   ├── meshes/                              # Piper STL collision & surface meshes
│   │   ├── piper_description.urdf               # URDF kinematic chain with corrected joint origins
│   │   ├── piper.urdf                           # Symlink to piper_description.urdf for URDF solvers
│   │   ├── piper_description.xml                # Native MuJoCo MJCF model with gripper_frame site
│   │   ├── robot.xml                            # Symlink to piper_description.xml
│   │   ├── scene.xml                            # MuJoCo world scene (lighting, floor, includes robot)
│   │   └── piper_description.xacro              # ROS Xacro macro source model
│   │
│   ├── so101_convert/                           # The Robot Studio SO-101 5-DOF arm assets
│   │   ├── meshes/                              # SO-101 STL 3D-printable meshes
│   │   ├── scene.xml                            # MuJoCo simulation environment scene
│   │   ├── scene_vrl.xml                        # Dedicated scene with overhead camera for Vision-RL
│   │   ├── so101.xml                            # Core MuJoCo MJCF robot model
│   │   └── so101_original.urdf                  # Original SO-101 URDF kinematic model
│   │
│   ├── ur5e/                                    # Universal Robots UR5e 6-DOF assets
│   │   ├── meshes/                              # Consolidated mesh repository (OBJ, STL, DAE)
│   │   │   ├── ur5e/                            # UR5e link visual DAE and collision STL meshes
│   │   │   └── robotiq_85_gripper/              # Robotiq 2F-85 gripper visual and collision meshes
│   │   ├── urdf/                                # UR5e URDF variants
│   │   │   ├── ur5e.urdf                        # UR5e URDF with calibrated 'pinch' fingertip link
│   │   │   ├── ur5e_with_gripper.urdf           # UR5e integrated with Robotiq 85 gripper
│   │   │   └── ur5e_with_camera.urdf            # UR5e with wrist-mounted RGB-D camera
│   │   ├── ur5e.urdf                            # Symlink to urdf/ur5e.urdf
│   │   ├── ur5e.xml                             # MuJoCo MJCF model of UR5e + 2F-85 gripper
│   │   ├── scene.xml                            # MuJoCo simulation scene with workspace desk & boxes
│   │   └── assets -> meshes                     # Symlink for MuJoCo relative meshdir resolution
│   │
│   ├── meshes -> Piper/meshes                   # Root fallback symlink for Piper mesh loading
│   └── so101_original.urdf -> so101_convert/... # Root fallback symlink for SO-101 URDF
│
├── solvers/                                     # Kinematic solver adapter implementations
│   ├── ikpy_solver.py                           # IKPy numerical least-squares solver adapter
│   ├── mink_solver.py                           # Mink differential QP solver adapter
│   ├── pinocchio_solver.py                      # Pinocchio CLIK damped least-squares adapter
│   └── pink_solver.py                           # Pink Pinocchio-based QP differential IK adapter
│
├── real_benchmark_results/                      # Single-folder storage for real hardware results
│   ├── real_results.csv                         # Master CSV log of real arm hardware telemetry
│   ├── comparison_report.md                     # Consolidated scorecard across hardware runs
│   └── solver_reports/                          # Individual solver hardware performance reports
│       ├── ikpy_report.md                       # Real-world IKPy execution telemetry
│       ├── mink_report.md                       # Real-world Mink execution telemetry
│       └── pinocchio_report.md                  # Real-world Pinocchio execution telemetry
│
├── sim_benchmark_results/                       # Single-folder storage for simulation results
│   ├── sim_results.csv                          # Master CSV log of simulation benchmark runs
│   ├── comparison_report.md                     # Consolidated scorecard across simulated runs
│   └── solver_reports/                          # Individual solver simulation performance reports
│       ├── ikpy_report.md                       # Simulated IKPy precision & latency report
│       ├── mink_report.md                       # Simulated Mink precision & latency report
│       ├── pinocchio_report.md                  # Simulated Pinocchio precision & latency report
│       └── pink_report.md                       # Simulated Pink precision & latency report
│
├── plots/                                       # Benchmark visualization graphics
│   ├── solve_time.png                           # Latency comparison boxplots across solvers
│   └── success_rate.png                         # Convergence success rate comparison charts
│
├── benchmark.py                                 # Core batch benchmark suite across targets.csv
├── common.py                                    # Kinematics evaluator, robot configs, and base classes
├── compare.py                                   # Evaluates Q1-Q4 engineering criteria and prints scorecard
├── custom_pose.txt                              # Preset test point definitions (pick, place, reach)
├── diagnose_failures.py                         # Singularity and reachability diagnostic analyzer
├── make_targets.py                              # Workspace random reachable target generator script
├── real_robot_benchmark.py                      # Interactive benchmark runner for real SO-101 arm
├── real_report.py                               # Hardware telemetry report generator utility
├── report_generator.py                          # Simulation report formatting engine
├── sim_live_viewer.py                           # 3D MuJoCo live viewer & interactive benchmark tool
├── targets.csv                                  # Database of reachable and edge-case Cartesian targets
├── target_3_pose.png                            # Reference pose render for Target 3
├── urdf_fk.py                                   # Pure NumPy forward kinematics model for validation
└── verify_models.py                             # Verification script checking MuJoCo MJCF compilations
```

---

## 7. Engineering Recommendations for Industry

1. **For Real-Time Visual Servoing & Teleoperation (>250 Hz):**
   - **Recommended Solver:** `PinocchioSolver`
   - **Reason:** Provides ultra-low solve latency (< 1.0 ms) while guaranteeing smooth CLIK trajectory integration.
2. **For Simulation & High-Precision Contact Manipulation:**
   - **Recommended Solver:** `MinkSolver`
   - **Reason:** Enforces hard joint configuration limits directly within the MuJoCo simulation environment, preventing self-collisions.
3. **For High-Precision Offline Trajectory Planning:**
   - **Recommended Solver:** `IKPySolver`
   - **Reason:** Yields near-zero Cartesian Euclidean position errors (< 0.001 mm) via global numerical optimization.

