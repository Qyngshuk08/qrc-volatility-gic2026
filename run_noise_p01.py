"""
Noise Sweep — p=0.01 only (standalone, run independently)
Run AFTER run_qubit_8.py has produced p=0 and p=0.001 rows in results/scaling_noise.csv.
This script appends/replaces only the p=0.01 row.

WARNING: default.mixed (density-matrix) is ~45x slower than default.qubit.
Expect ~50-60 min for 300 train + 150 test samples at N=8.

Usage: python run_noise_p01.py
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from qrc_pipeline import load_data, preprocess, split, build_qrc, extract_all, CFG, log

N_QUBITS = 8
N_TRAIN  = 300
N_TEST   = 150
NOISE_P  = 0.01
CSV_PATH = "results/scaling_noise.csv"

def main():
    log(f"=== Noise sweep: p={NOISE_P} (N={N_QUBITS}, subsampled {N_TRAIN}/{N_TEST}) ===")
    log("WARNING: density-matrix sim is ~45x slower than statevector. Expect 50-60 min.")

    rv_series, source = load_data()
    X_raw, X_har, X_harj, y, dates = preprocess(rv_series, CFG)
    (tr_raw,_,_,tr_y,_), _, (te_raw,_,_,te_y,_) = split(X_raw, X_har, X_harj, y, dates, CFG)

    tr_raw, tr_y = tr_raw[-N_TRAIN:], tr_y[-N_TRAIN:]
    te_raw, te_y = te_raw[:N_TEST],   te_y[:N_TEST]

    cfg = dict(CFG); cfg["n_qubits"] = N_QUBITS
    g_mean = float(np.mean(tr_raw)); g_std = float(np.std(tr_raw)) + 1e-8

    t0 = time.time()
    extractor, _ = build_qrc(cfg, noise_p=NOISE_P, global_mean=g_mean, global_std=g_std)
    Xq_tr = extract_all(tr_raw, extractor, f"p={NOISE_P} Train")
    Xq_te = extract_all(te_raw, extractor, f"p={NOISE_P} Test")

    sc = StandardScaler()
    m  = RidgeCV(alphas=CFG["ridge_alphas"])
    m.fit(sc.fit_transform(Xq_tr), tr_y)
    pred = m.predict(sc.transform(Xq_te))
    elapsed = time.time() - t0

    rmse = float(np.sqrt(mean_squared_error(te_y, pred)))
    log(f"RESULT: p={NOISE_P} RMSE={rmse:.4f} time={elapsed:.1f}s alpha={m.alpha_}")

    row = {"n_qubits": N_QUBITS, "noise_p": NOISE_P, "RMSE": round(rmse,4),
           "ridge_alpha": float(m.alpha_), "wall_clock_s": round(elapsed,1)}

    # Append/replace this p-row in the shared CSV without touching other rows
    df_new = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df_old = df_old[df_old["noise_p"] != NOISE_P]
        df = pd.concat([df_old, df_new], ignore_index=True).sort_values("noise_p")
    else:
        df = df_new
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved to {CSV_PATH}")

if __name__ == "__main__":
    main()
