"""
Qubit Scaling — N=15 (standalone script, run independently)

WARNING: statevector simulation cost scales as 2^N. N=15 is ~100-250x more
expensive per sample than N=8. Dataset is cut to 60 train / 30 test to keep
this survivable in a single 2hr qBraid session. If this still times out,
report N=15 as a projected (not measured) data point in the write-up and
disclose this limitation explicitly — that is a legitimate, honest finding.

Usage: python run_qubit_15.py
Appends one row to results/scaling_qubit.csv
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

N_QUBITS   = 15
N_TRAIN    = 60    # heavily cut — 2^15 statevector is ~100-250x cost of N=8
N_TEST     = 30
CSV_PATH   = "results/scaling_qubit.csv"

def main():
    log(f"=== Qubit scaling: N={N_QUBITS} (heavily subsampled {N_TRAIN}/{N_TEST}) ===")
    log("WARNING: this qubit count is expensive. If it stalls, Ctrl+C and")
    log("report N=15 as a projected data point with this limitation disclosed.")

    rv_series, source = load_data()
    X_raw, X_har, X_harj, y, dates = preprocess(rv_series, CFG)
    (tr_raw,_,_,tr_y,_), _, (te_raw,_,_,te_y,_) = split(X_raw, X_har, X_harj, y, dates, CFG)

    tr_raw, tr_y = tr_raw[-N_TRAIN:], tr_y[-N_TRAIN:]
    te_raw, te_y = te_raw[:N_TEST],   te_y[:N_TEST]

    cfg = dict(CFG); cfg["n_qubits"] = N_QUBITS
    g_mean = float(np.mean(tr_raw)); g_std = float(np.std(tr_raw)) + 1e-8

    t0 = time.time()
    extractor, _ = build_qrc(cfg, noise_p=0.0, global_mean=g_mean, global_std=g_std)
    Xq_tr = extract_all(tr_raw, extractor, f"N={N_QUBITS} Train")
    Xq_te = extract_all(te_raw, extractor, f"N={N_QUBITS} Test")

    sc = StandardScaler()
    m  = RidgeCV(alphas=CFG["ridge_alphas"])
    m.fit(sc.fit_transform(Xq_tr), tr_y)
    pred = m.predict(sc.transform(Xq_te))
    elapsed = time.time() - t0

    rmse = float(np.sqrt(mean_squared_error(te_y, pred)))
    row = {"n_qubits": N_QUBITS, "noise_p": 0.0, "RMSE": round(rmse,4),
           "ridge_alpha": float(m.alpha_), "wall_clock_s": round(elapsed,1),
           "circuit_depth": 3*N_QUBITS, "n_train": N_TRAIN, "n_test": N_TEST,
           "note": "heavily subsampled due to 2^N statevector cost"}
    log(f"RESULT: N={N_QUBITS} RMSE={rmse:.4f} time={elapsed:.1f}s alpha={m.alpha_}")

    df_new = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df_old = df_old[df_old["n_qubits"] != N_QUBITS]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved to {CSV_PATH}")

if __name__ == "__main__":
    main()
