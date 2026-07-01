# HF-QRC: Hybrid Feedback Quantum Reservoir Computing for Volatility Forecasting

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com?gitHubUrl=https://github.com/YOUR_GITHUB_USERNAME/qrc-volatility-gic2026)

**Team:** Qudit Creons  
**Challenge:** qBraid × MITRE × JonesTrading | GIC 2026  
**Track:** Track A — Financial Volatility Prediction  
**Sub-problem:** 5-day-ahead S&P 500 Realized Volatility Forecasting  
**Phase:** 3 (Final)  
**Deadline:** July 26, 2026

---

## Project Overview

We implement a **Hybrid Feedback Quantum Reservoir Computer (HF-QRC)** for forecasting S&P 500 realized volatility at the 5-day horizon. The reservoir is a gate-based Transverse-Field Ising Model (TFIM) with all-to-all connectivity, angle encoding, and a measurement feedback loop. A ridge-regression readout layer is trained on quantum feature vectors extracted from the reservoir.

**Key result:** HF-QRC (N=8 qubits) outperforms all classical baselines including HAR, HAR-J, ESN-50/100, GARCH(1,1), and Persistence on RMSE, MAE, and QLIKE metrics on the held-out test set (Jan 2020 – Dec 2022, including COVID-19 stress period).

---

## Repository Structure

```
QuditCreons_QRC_Phase3/
├── qrc_pipeline.py       # Main pipeline: data → QRC features → baselines → metrics
├── noise_scaling.py      # Qubit scaling (N=5,8,10,15) + noise study (p=0→0.05)
├── qpu_run.py            # QPU validation on IBM Quantum (Qiskit)
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── results/              # Auto-generated on run
    ├── metrics_table.csv     # Full benchmark table
    ├── predictions.csv       # Full prediction time series
    ├── scaling_qubit.csv     # RMSE vs qubit count
    ├── scaling_noise.csv     # RMSE vs noise level
    ├── scaling_summary.txt   # Scaling tables for write-up
    ├── qpu_results.json      # QPU hardware run results
    └── run_log.txt           # Wall-clock times, configs, logs
```

---

## Setup Instructions

### Option A: Run on qBraid Lab (Recommended — click Launch on qBraid above)

1. Click the **Launch on qBraid** button at the top of this README
2. This clones the repository into your qBraid Lab environment
3. Open a terminal in qBraid Lab and run:

```bash
pip install -r requirements.txt -q
```

4. (Optional) Download OMI-RV dataset for best results:
   - Go to: https://realized.oxford-man.ox.ac.uk/data/download
   - Download `oxfordmanrealizedvolatilityindices.csv`
   - Upload to the same directory as the scripts
   - If not present, the pipeline automatically uses Yahoo Finance Parkinson RV

5. Run the pipeline (see **Running the Code** below)

### Option B: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/qrc-volatility-gic2026
cd qrc-volatility-gic2026

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
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

Install all at once:
```bash
pip install -r requirements.txt -q
```

All packages are pre-installed in qBraid Lab. No external configuration required.

---

## Running the Code

### Step 1 — Main Pipeline (required)

Runs full HF-QRC vs. all classical baselines on the volatility forecasting task.

```bash
python qrc_pipeline.py
```

**Expected runtime:** 15–35 minutes on qBraid CPU (N=8, T=8, ~4,700 training samples)  
**Outputs:** `results/metrics_table.csv`, `results/predictions.csv`, `results/run_log.txt`

**Expected output (example):**
```
  Model                  RMSE     MAE    QLIKE   MZ-R²
  -------------------------------------------------------
  HF-QRC (8q)          0.1421  0.1052  0.2187  0.6841  ◄ QRC
  HAR                  0.1698  0.1284  0.2931  0.5612
  HAR-J                0.1653  0.1241  0.2814  0.5788
  ESN (50 nodes)       0.1579  0.1188  0.2643  0.6023
  ESN (100 nodes)      0.1542  0.1163  0.2571  0.6198
  GARCH(1,1)           0.1847  0.1391  0.3214  0.5104
  Persistence          0.2134  0.1623  0.4012  0.4321
```

### Step 2 — Qubit Scaling + Noise Study (required)

```bash
python noise_scaling.py
```

**Expected runtime:** 45–90 minutes (runs N=5,8,10,15 × noise levels)  
**Outputs:** `results/scaling_qubit.csv`, `results/scaling_noise.csv`, `results/scaling_summary.txt`

### Step 3 — QPU Validation Run (optional, requires IBM token)

```bash
# Set your IBM Quantum token (get from quantum.ibm.com → API token)
export IBM_TOKEN="your_token_here"
python qpu_run.py
```

**If no IBM token:** script runs in statevector demo mode automatically — no error.  
**Expected runtime:** 5–30 minutes depending on IBM queue length  
**Outputs:** `results/qpu_results.json`, `results/qpu_summary.txt`

---

## Expected Inputs / Outputs

| Script | Input | Output |
|--------|-------|--------|
| `qrc_pipeline.py` | OMI-RV CSV (optional) or Yahoo Finance (auto) | `results/metrics_table.csv` |
| `noise_scaling.py` | Same as above (auto) | `results/scaling_*.csv` |
| `qpu_run.py` | `IBM_TOKEN` env var (optional) | `results/qpu_results.json` |

---

## Reproducing the Write-Up Numbers

After running `qrc_pipeline.py`, open `results/metrics_table.csv`. The numbers in Table 1 of the write-up come directly from this file.

After running `noise_scaling.py`, open `results/scaling_summary.txt`. Tables 2 and 3 in the write-up come from this file.

After running `qpu_run.py`, open `results/qpu_summary.txt` for the hardware validation numbers in Section 5 of the write-up.

---

## Architecture Summary

| Component | Design | Justification |
|-----------|--------|---------------|
| Hamiltonian | All-to-all TFIM, J~N(0,1), h=0.5 | Maximises IPC at quantum critical point (Cindrak et al. 2026) |
| Encoding | Angle Ry(π·tanh(x)), T=8 temporal multiplexing | 22-lag effective memory, no look-ahead bias |
| Observables | ⟨σ_z^i⟩ + ⟨σ_z^i σ_z^j⟩, 36 total (N=8) | Implicit HAR-J cross-frequency features |
| Feedback | β·y_prev → Ry on qubit N-1 | Restores fading memory (Zhu et al. 2025) |
| Readout | Ridge regression + degree-2 poly | Nonlinear features at zero qubit cost |

---

## Known Limitations

1. **Statevector simulation scales as 2^N memory.** N=15 requires ~1GB RAM; N=20 requires ~32GB. On qBraid CPU instances, we recommend N≤12 for statevector. Use density-matrix (`default.mixed`) only for noise studies at N≤10.

2. **Yahoo Finance Parkinson RV ≠ OMI-RV rv5.** Results are best reproduced with the OMI-RV dataset. Yahoo Finance fallback gives directionally consistent but numerically different results. For exact write-up numbers, download OMI-RV.

3. **QPU results depend on queue length and backend calibration.** IBM backend selection is automatic (least-busy). Hardware noise levels vary by calibration date.

4. **Ridge regression readout is linear.** Our polynomial expansion (degree 2) partially addresses this, but deep nonlinear readouts (e.g., small MLP) may further improve performance at the cost of interpretability.

5. **T=8 temporal multiplexing.** Increasing T improves memory capacity but increases wall-clock time linearly. T=8 was chosen as the best trade-off on the validation set.

---

## References

1. Kornjaca et al., arXiv:2407.02553 (2024)
2. Zhu et al., Phys. Rev. Research 7, 023290 (2025)
3. Ahmed et al., Proc. R. Soc. A 481, 20250550 (2025)
4. Li et al., arXiv:2505.13933 (2025)
5. Cindrak et al., arXiv:2603.21371 (2026)
6. Antoncich et al., arXiv:2602.14641 (2026)
7. Hou et al., Phys. Rev. Lett. 136, 120602 (2026)
8. Patton, J. Econometrics 160, 246-256 (2011)
9. Corsi, J. Financ. Econometrics 7(2), 174-196 (2009)

---

## Contact

**Abhishek Raj** | Aqora: @qyngshuq | kvt057@gmail.com  
Team: Qudit Creons | GIC 2026
