"""Generate a PDF/HTML report from real_results.csv
Usage:  python real_report.py
Saves:  real_report/report.html   (always)
        real_report/report.pdf    (if weasyprint installed)
"""
import sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
CSV  = HERE / "real_results.csv"
OUT  = HERE / "real_report"

try:
    import pandas as pd
except ImportError:
    print("pip install pandas"); sys.exit(1)

def main():
    if not CSV.exists():
        print(f"No {CSV} found. Run real_robot_benchmark.py first.")
        sys.exit(1)

    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(CSV)
    df = df[df["notes"] != "SAFETY_FAIL_skipped"]  # exclude unsafe-skipped rows

    # ── Per-solver summary ────────────────────────────────────────────────
    grp = df.groupby("solver").agg(
        n_trials          = ("target_id",        "count"),
        ik_solve_ms_med   = ("ik_solve_ms",      "median"),
        sim_pos_err_mm    = ("sim_pos_err_mm",   "median"),
        real_pos_err_mm   = ("real_pos_err_mm",  "median"),
        joint_dist_deg    = ("joint_dist_deg",   "median"),
        cartesian_dist_mm = ("cartesian_dist_mm","median"),
        mean_torque_pct   = ("mean_torque_pct",  "mean"),
        mean_power_mW     = ("approx_power_mW",  "mean"),
        settle_time_ms    = ("settle_time_ms",   "median"),
    ).reset_index()

    # ── Best-per-metric ───────────────────────────────────────────────────
    best = {
        "Fastest IK":          grp.loc[grp.ik_solve_ms_med.idxmin(),   "solver"],
        "Lowest real pos err": grp.loc[grp.real_pos_err_mm.idxmin(),   "solver"],
        "Lowest torque":       grp.loc[grp.mean_torque_pct.idxmin(),   "solver"],
        "Lowest power":        grp.loc[grp.mean_power_mW.idxmin(),     "solver"],
        "Fastest settle":      grp.loc[grp.settle_time_ms.idxmin(),    "solver"],
    }

    # ── HTML report ───────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>SO-101 Real-Robot IK Benchmark Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1   {{ color: #1a1a6e; }}
  h2   {{ color: #2a2a8e; border-bottom: 1px solid #aaa; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
  th   {{ background: #2a2a8e; color: white; padding: 6px 10px; }}
  td   {{ border: 1px solid #ccc; padding: 5px 10px; }}
  tr:nth-child(even) {{ background: #f0f4ff; }}
  .best {{ font-weight: bold; color: #006600; }}
  .note {{ font-size: 12px; color: #555; }}
</style>
</head><body>
<h1>SO-101 Real-Robot IK Benchmark Report</h1>
<p class="note">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; Trials: {len(df)}</p>

<h2>1. Per-Solver Summary (median over all trials)</h2>
{grp.round(2).to_html(index=False, classes="", border=0)}

<h2>2. Best Solver Per Metric</h2>
<table>
<tr><th>Metric</th><th>Best Solver</th></tr>
{''.join(f"<tr><td>{k}</td><td class='best'>{v}</td></tr>" for k,v in best.items())}
</table>

<h2>3. Metric Definitions</h2>
<table>
<tr><th>Column</th><th>Meaning</th></tr>
<tr><td>ik_solve_ms_med</td><td>Median IK computation time (ms) — pure software speed</td></tr>
<tr><td>sim_pos_err_mm</td><td>IK solution error in simulation (MuJoCo FK) — solver accuracy</td></tr>
<tr><td>real_pos_err_mm</td><td>Actual arm FK vs target after settling — mechanical accuracy</td></tr>
<tr><td>joint_dist_deg</td><td>Sum of |Δq| from home to goal (deg) — how much the arm moved</td></tr>
<tr><td>cartesian_dist_mm</td><td>Straight-line distance from start EE pos to target (mm)</td></tr>
<tr><td>mean_torque_pct</td><td>Mean motor duty cycle while moving (0–100%) — proxy for torque</td></tr>
<tr><td>mean_power_mW</td><td>Approx. mean power = V × (load% × rated_current). Not measured directly.</td></tr>
<tr><td>settle_time_ms</td><td>Time until joint velocities dropped below 0.5°/sample</td></tr>
</table>

<h2>4. Raw Data</h2>
{df.round(3).to_html(index=False, classes="", border=0)}

<p class="note">
⚠ Power values are approximate (load% × rated_current × bus_voltage). STS3215 does not expose a direct current register.
Joint velocity jerk is estimated from {1000//50}-Hz position polling, not from encoder velocity registers.
</p>
</body></html>"""

    html_path = OUT / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved → {html_path}")

    # Try PDF
    try:
        from weasyprint import HTML
        pdf_path = OUT / "report.pdf"
        HTML(string=html).write_pdf(str(pdf_path))
        print(f"PDF  report saved → {pdf_path}")
    except ImportError:
        print("(pip install weasyprint for PDF output)")
    except Exception as e:
        print(f"(PDF failed: {e})")

    # Print to terminal
    print("\n" + "="*70)
    print("REAL-ROBOT BENCHMARK — SUMMARY TABLE")
    print("="*70)
    cols_show = ["solver","n_trials","ik_solve_ms_med","sim_pos_err_mm",
                 "real_pos_err_mm","joint_dist_deg","mean_torque_pct","mean_power_mW","settle_time_ms"]
    print(grp[cols_show].round(2).to_string(index=False))
    print("\nBest per metric:")
    for k, v in best.items():
        print(f"  {k:<25}: {v}")

if __name__ == "__main__":
    main()

