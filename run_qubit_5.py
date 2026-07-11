"""
Qubit Scaling — N=5 (standalone script, run independently)
Usage: python run_qubit_5.py
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

N_QUBITS   = 5
N_TRAIN    = 300   # subsampled — full 4501 is too slow for a 2hr session
N_TEST     = 150
CSV_PATH   = "results/scaling_qubit.csv"

def main():
    log(f"=== Qubit scaling: N={N_QUBITS} (subsampled {N_TRAIN}/{N_TEST}) ===")
    rv_series, source = load_data()
    X_raw, X_har, X_harj, y, dates = preprocess(rv_series, CFG)
    (tr_raw,_,_,tr_y,_), _, (te_raw,_,_,te_y,_) = split(X_raw, X_har, X_harj, y, dates, CFG)

    tr_raw, tr_y = tr_raw[-N_TRAIN:], tr_y[-N_TRAIN:]   # most recent N_TRAIN (avoid stale regime)
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
           "circuit_depth": 3*N_QUBITS, "n_train": N_TRAIN, "n_test": N_TEST}
    log(f"RESULT: N={N_QUBITS} RMSE={rmse:.4f} time={elapsed:.1f}s alpha={m.alpha_}")

    # Append (create if not exists)
    df_new = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df_old = df_old[df_old["n_qubits"] != N_QUBITS]  # replace if rerun
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved to {CSV_PATH}")

if __name__ == "__main__":
    main()
