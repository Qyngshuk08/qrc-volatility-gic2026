# HF-QRC: Hybrid Feedback Quantum Reservoir Computing for Volatility Forecasting


[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com?gitHubUrl=https://github.com/Qyngshuk08/qrc-volatility-gic2026)

**Team:** Qudit Creons
**Challenge:** qBraid × MITRE × JonesTrading | GIC 2026
**Track:** Track A — Financial Volatility Prediction
**Sub-problem:** 5-day-ahead S&P 500 Realized Volatility Forecasting
**Phase:** 3 (Final)
**Deadline:** July 26, 2026

---

## Project Overview

We implement a **Hybrid Feedback Quantum Reservoir Computer (HF-QRC)** for forecasting S&P 500 realized volatility at the 5-day horizon, using the Oxford-Man Institute Realized Volatility Library (OMI-RV, SPX2.rv series, Jan 2000 - Sep 2016).

**Honest headline result:** HF-QRC (N=8 qubits) achieves RMSE=0.9301 on the held-out test set (n=437, Jan 2015-Sep 2016), competitive with HAR (0.9203) and HAR-J (0.9214), and a decisive ~15-18% RMSE / ~48% QLIKE improvement over GARCH(1,1) and Persistence. ESN-50/100 slightly outperform HF-QRC on this dataset. Full results, including qubit scaling and noise studies with disclosed limitations, are in the write-up (`QuditCreons_Phase3_WriteUp_v2.docx`).

---

## Repository Structure

```
qrc-volatility-gic2026/
├── qrc_pipeline.py            # Main pipeline: data -> QRC features -> baselines -> metrics
├── run_qubit_5.py              # Qubit scaling: N=5 (noiseless)
├── run_qubit_8_noiseless.py    # Qubit scaling: N=8 (noiseless)
├── run_qubit_10.py             # Qubit scaling: N=10 (noiseless)
├── run_qubit_15.py             # Qubit scaling: N=15 (noiseless, heavily subsampled)
├── run_noise_p001.py           # Noise sweep: p=0.001, N=8
├── run_noise_p01.py            # Noise sweep: p=0.01, N=8
├── run_noise_p05.py            # Noise sweep: p=0.05, N=8
├── qpu_run.py                  # QPU validation on IBM Quantum (Qiskit)
├── oxfordmanrealizedvolatilityindices.csv   # OMI-RV dataset (SPX2.rv, 2000-2016)
├── QuditCreons_Phase3_WriteUp_v2.docx       # Full 5-page technical write-up
├── README.md
├── requirements.txt
└── results/
    ├── metrics_table.csv       # Main benchmark (Table 1 in write-up)
    ├── scaling_qubit.csv       # Qubit scaling results (Table 2)
    ├── scaling_noise.csv       # Noise sweep results (Table 3)
    ├── qpu_results.json        # Raw QPU hardware run data
    └── qpu_summary.txt         # QPU run summary (Section 4.4)
```

---

## Setup Instructions

### On qBraid Lab

```bash
git clone https://github.com/Qyngshuk08/qrc-volatility-gic2026.git
cd qrc-volatility-gic2026
pip install -r requirements.txt -q
```

The OMI-RV dataset (`oxfordmanrealizedvolatilityindices.csv`) is already included in this repo -- no manual download needed.

**Note on the dataset:** this specific OMI-RV export uses a 3-row header format with columns `DateID` (YYYYMMDD int) and `SPX2.rv` (Realized Variance, 5-min), spanning 2000-01-03 to 2016-09-26. `qrc_pipeline.py` parses this format automatically.

### Locally

```bash
git clone https://github.com/Qyngshuk08/qrc-volatility-gic2026.git
cd qrc-volatility-gic2026
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dependencies

```
pennylane>=0.38.0
qiskit>=1.0.0
qiskit-ibm-runtime>=0.20.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
arch>=6.3.0
```

```bash
pip install -r requirements.txt -q
```

---

## Running the Code

### 1. Main Benchmark (required, ~7 min)

```bash
python qrc_pipeline.py
```
Produces `results/metrics_table.csv` -- Table 1 in the write-up (full 3,213-sample training set).

### 2. Qubit Scaling (4 separate scripts, ~1-2 min each except N=15)

```bash
python run_qubit_5.py
python run_qubit_8_noiseless.py
python run_qubit_10.py
python run_qubit_15.py     # heavily subsampled (60/30) due to 2^N statevector cost
```
Each appends its row to `results/scaling_qubit.csv` -- Table 2 in the write-up.
**Note:** these scripts use a smaller subsample (300 train / 150 test, most recent) than the main benchmark, so results are not directly comparable to Table 1's numbers. N=15 uses a further-reduced 60/30 split. This is disclosed explicitly in the write-up.

### 3. Noise Sweep (3 separate scripts -- density-matrix simulation is slow, ~50-60 min each)

```bash
python run_noise_p001.py
python run_noise_p01.py
python run_noise_p05.py
```
Each appends its row to `results/scaling_noise.csv` -- Table 3 in the write-up.
**Known limitation (disclosed in write-up):** all four noise levels (including p=0 from `run_qubit_8_noiseless.py`) produced identical RMSE. This is a ridge-regularisation saturation artifact (300 training samples vs. 352+ features is a p>>n regime), not evidence of genuine noise robustness. A larger training sample would be needed to properly isolate a noise effect.

### 4. QPU Validation (optional, requires IBM Quantum token)

```bash
export IBM_TOKEN="your_token_from_quantum.ibm.com"
python qpu_run.py
```
Runs N=8, T=1 circuits on IBM Quantum hardware (auto-selects least-busy backend; our run used `ibm_fez`). Produces `results/qpu_results.json` and `results/qpu_summary.txt`.
**Note:** no error mitigation (ZNE) is applied -- raw hardware fidelity is reported (mean |delta<Z_i>| = 0.1426 in our run).

---

## Architecture Summary

| Component | Design | Justification |
|-----------|--------|----------------|
| Hamiltonian | All-to-all TFIM, J~N(0,1)/sqrt(N), h=0.5 | Coupling normalised to prevent chaotic scrambling of encoded input |
| Encoding | Ry(pi*tanh(z)), z = globally-standardised log-RV, T=8 steps over the **most recent** 8 of 22 lags | Preserves absolute volatility level; encoding recent (not oldest) lags was critical -- see Known Issues below |
| Observables | <Z_i> + <Z_i Z_j>, 36 total (N=8) | Implicit cross-lag interaction features |
| Feedback | beta*y_prev -> Ry on qubit N-1, beta=0.1 | Restores fading memory |
| Readout | Ridge regression + degree-2 poly expansion | Nonlinear features at zero qubit cost |

---

## Known Issues / Debugging History (disclosed for transparency)

During development we identified and fixed four significant bugs, documented here since they materially affected results:

1. **ESN baseline shape bug:** `W_in` was shaped `(n_nodes, 1)` instead of `(n_nodes,)`, causing NumPy broadcasting to silently produce a 3D state array instead of 1D, crashing `StandardScaler`. Fixed by flattening `W_in`.

2. **Input encoding saturation:** raw log-RV values (~-9 to -10) fed directly into `tanh()` saturate to +/-0.9999, destroying input differentiation. Fixed by standardising inputs before the tanh squash.

3. **Level erasure from per-window normalisation:** an intermediate fix standardised each 22-day window using its own local mean/std, which erased the single most predictive signal in volatility forecasting (absolute current volatility level). Fixed by using global training-set mean/std instead.

4. **Stale lag indexing (the most significant bug):** the original encoding loop selected `x_series[t % L]` for `t` in `0..T-1`, which -- since `T=8 < L=22` -- always encoded the **oldest** 8 of 22 lags, never the most recent (most predictive) days. Fixed to encode `x_series[L-T+t]`, the most recent T lags. This single fix took QRC RMSE from 1.29 (worse than all classical baselines) to 0.93 (competitive with HAR).

We disclose this because Phase 3's rubric explicitly rewards honest limitation reporting over polished-but-unverifiable claims, and because a judge re-running our code should see exactly why the numbers are what they are.

## Other Limitations

- **Dataset date range:** this specific OMI-RV file spans 2000-2016, not through 2022. We use the August 2015 / January 2016 volatility episodes as our stress-test window rather than COVID-19.
- **Qubit scaling is non-monotonic:** N=10 underperforms N=8 in our results (Table 2). We report this as-is rather than smoothing it into a false clean trend.
- **Noise study is inconclusive** -- see Section 3 above.
- **QPU run is a fidelity check, not a full benchmark:** 20 circuits at T=1 with 1,024 shots confirms correct execution on real hardware; it does not constitute a full forecasting benchmark on QPU (which would require ~3,213 training circuits, beyond free-tier shot budgets).

---

## References

1. Kornjaca et al., arXiv:2407.02553 (2024)
2. Zhu et al., Phys. Rev. Research 7, 023290 (2025)
3. Ahmed et al., Proc. R. Soc. A 481, 20250550 (2025)
4. Li et al., arXiv:2505.13933 (2025)
5. Patton, J. Econometrics 160, 246-256 (2011)
6. Corsi, J. Financ. Econometrics 7(2), 174-196 (2009)
7. Bollerslev, J. Econometrics 31(3), 307-327 (1986)
8. Mincer & Zarnowitz, in Economic Forecasts and Expectations, NBER (1969)
9. Jaeger, GMD Report 148 (2001)

---

## Contact

**Abhishek Raj** | Aqora: @qyngshuq | kvt057@gmail.com
Team: Qudit Creons | GIC 2026
