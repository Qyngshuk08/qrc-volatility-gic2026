"""
=============================================================================
HF-QRC Noise & Qubit Scaling Experiment — Qudit Creons | GIC 2026
=============================================================================
Characterises how QRC performance scales with:
  1. Qubit count N = 5, 8, 10, 15
  2. Depolarising noise p = 0, 0.001, 0.01, 0.05

Run AFTER qrc_pipeline.py (reuses data loading & preprocessing).

Usage:
  python noise_scaling.py

Outputs:
  results/scaling_qubit.csv   — RMSE vs N (noiseless)
  results/scaling_noise.csv   — RMSE vs noise p (N=8 fixed)
  results/scaling_summary.txt — table for write-up
=============================================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# Re-use data loading from pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from qrc_pipeline import (
    load_data, preprocess, split, build_qrc, extract_all,
    compute_metrics, CFG, log
)

QUBIT_COUNTS  = [5, 8, 10, 15]
NOISE_LEVELS  = [0.0, 0.001, 0.01, 0.05]
RIDGE_ALPHAS  = CFG["ridge_alphas"]


def fit_predict(Xtr, Xte, ytr):
    sc = StandardScaler()
    m  = RidgeCV(alphas=RIDGE_ALPHAS)
    m.fit(sc.fit_transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), float(m.alpha_)


def run_experiment(X_raw_tr, X_raw_te, tr_y, te_y, n_qubits, noise_p):
    cfg = dict(CFG)
    cfg["n_qubits"] = n_qubits

    t0 = time.time()
    extractor, _ = build_qrc(cfg, noise_p=noise_p)
    Xq_tr = extract_all(X_raw_tr, extractor, f"N={n_qubits},p={noise_p} Train")
    Xq_te = extract_all(X_raw_te, extractor, f"N={n_qubits},p={noise_p} Test")
    pred, alpha = fit_predict(Xq_tr, Xq_te, tr_y)
    elapsed = time.time() - t0

    r = float(np.sqrt(mean_squared_error(te_y, pred)))
    return {
        "n_qubits":    n_qubits,
        "noise_p":     noise_p,
        "RMSE":        round(r, 4),
        "ridge_alpha": alpha,
        "wall_clock_s": round(elapsed, 1),
        "circuit_depth": 3 * n_qubits,
    }


def main():
    log("=" * 65)
    log("HF-QRC Noise & Qubit Scaling — Qudit Creons | GIC 2026")
    log("=" * 65)

    # Load & preprocess once
    log("\n[1/3] Loading data...")
    rv_series, source = load_data()
    log(f"Data: {source}")
    X_raw, X_har, X_harj, y, dates = preprocess(rv_series, CFG)
    (tr_raw,_,_,tr_y,_), _, (te_raw,_,_,te_y,_) = split(
        X_raw, X_har, X_harj, y, dates, CFG)

    # ── Qubit scaling (noiseless) ─────────────────────────────────────────
    log("\n[2/3] Qubit scaling study (noise_p=0)...")
    qubit_rows = []
    for N in QUBIT_COUNTS:
        log(f"\n  Running N={N} qubits...")
        row = run_experiment(tr_raw, te_raw, tr_y, te_y, N, 0.0)
        qubit_rows.append(row)
        log(f"  N={N}: RMSE={row['RMSE']:.4f}, depth={row['circuit_depth']}, "
            f"time={row['wall_clock_s']}s")

    qubit_df = pd.DataFrame(qubit_rows)
    qubit_df.to_csv("results/scaling_qubit.csv", index=False)
    log("\n  Saved: results/scaling_qubit.csv")

    # ── Noise scaling (N=8 fixed) ─────────────────────────────────────────
    log("\n[3/3] Noise scaling study (N=8 fixed)...")
    noise_rows = []
    for p in NOISE_LEVELS:
        log(f"\n  Running noise_p={p}...")
        row = run_experiment(tr_raw, te_raw, tr_y, te_y, 8, p)
        noise_rows.append(row)
        log(f"  p={p}: RMSE={row['RMSE']:.4f}, time={row['wall_clock_s']}s")

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv("results/scaling_noise.csv", index=False)
    log("\n  Saved: results/scaling_noise.csv")

    # ── Summary table ─────────────────────────────────────────────────────
    summary = []
    summary.append("=" * 60)
    summary.append("QUBIT SCALING (noiseless, depolarising p=0)")
    summary.append(f"  {'N':>4}  {'RMSE':>7}  {'Depth':>6}  {'Time(s)':>8}")
    summary.append("  " + "-" * 30)
    for row in qubit_rows:
        summary.append(f"  {row['n_qubits']:>4}  {row['RMSE']:>7.4f}  "
                        f"{row['circuit_depth']:>6}  {row['wall_clock_s']:>8.1f}")
    summary.append("")
    summary.append("NOISE SCALING (N=8 fixed, depolarising channel)")
    summary.append(f"  {'p':>7}  {'RMSE':>7}  {'Time(s)':>8}")
    summary.append("  " + "-" * 25)
    for row in noise_rows:
        summary.append(f"  {row['noise_p']:>7.3f}  {row['RMSE']:>7.4f}  "
                        f"{row['wall_clock_s']:>8.1f}")
    summary.append("=" * 60)

    summary_str = "\n".join(summary)
    print("\n" + summary_str)
    with open("results/scaling_summary.txt", "w") as f:
        f.write(summary_str)
    log("\n  Saved: results/scaling_summary.txt")

    return qubit_df, noise_df


if __name__ == "__main__":
    main()
