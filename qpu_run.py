"""
=============================================================================
HF-QRC QPU Validation Run — Qudit Creons | GIC 2026
=============================================================================
Runs a small N=8, T=1 QRC circuit on a real QPU via Qiskit.
Uses IBM Quantum Open Plan (free, no approval needed).

This script:
  1. Builds the TFIM QRC circuit in Qiskit
  2. Runs it on ibm_brisbane (or least-busy 127q IBM backend)
  3. Compares shot-based expectation values vs. statevector simulation
  4. Reports hardware vs. noiseless RMSE gap (noise effect analysis)

Setup:
  pip install qiskit qiskit-ibm-runtime
  Set IBM_TOKEN environment variable:
    export IBM_TOKEN="your_ibm_quantum_token"
  Or paste token directly below (TOKEN variable).

Usage:
  python qpu_run.py

Outputs:
  results/qpu_results.json   — shot-based vs. statevector comparison
  results/qpu_summary.txt    — table for write-up
=============================================================================
"""

import os, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── IBM Token ─────────────────────────────────────────────────────────────────
# Set via environment variable (recommended) or paste here
IBM_TOKEN = os.environ.get("IBM_TOKEN", "PASTE_YOUR_IBM_TOKEN_HERE")

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS   = 8       # reservoir size
T_STEPS    = 1       # single step for QPU (keep circuit shallow)
SHOTS      = 1024    # shot budget per circuit
N_SAMPLES  = 20      # number of test input samples to run on QPU
SEED       = 42
H_FIELD    = 0.5     # transverse field


def build_tfim_circuit(x_in, feedback, n_qubits, h_field, J):
    """
    Build a single TFIM QRC step as a Qiskit QuantumCircuit.

    Args:
        x_in:     scalar input value (log-RV)
        feedback: scalar feedback value from previous step
        n_qubits: number of qubits
        h_field:  transverse field strength
        J:        coupling matrix (n_qubits x n_qubits)

    Returns:
        qc: QuantumCircuit with measurements
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)

    # ── Input encoding ────────────────────────────────────────────────────
    enc_angle = float(np.pi * np.tanh(x_in))
    fb_angle  = float(np.pi * np.tanh(0.1 * feedback))

    for i in range(n_qubits - 1):
        qc.ry(enc_angle + 0.1 * i * x_in, i)
    qc.ry(fb_angle, n_qubits - 1)

    # ── TFIM evolution (Trotterised, 1 step) ─────────────────────────────
    # ZZ interactions
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if abs(J[i, j]) > 0.01:  # prune weak couplings
                qc.rzz(2 * J[i, j], i, j)

    # Transverse field (X rotations)
    for i in range(n_qubits):
        qc.rx(2 * h_field, i)

    # ── Measure all qubits ─────────────────────────────────────────────
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


def shots_to_expval(counts, qubit_idx, n_qubits):
    """
    Convert shot counts to <Z_i> expectation value.
    Z eigenvalue: +1 for |0>, -1 for |1>
    """
    total = sum(counts.values())
    expval = 0.0
    for bitstring, count in counts.items():
        # Qiskit bitstring: rightmost bit = qubit 0
        bits = bitstring.zfill(n_qubits)
        bit  = int(bits[n_qubits - 1 - qubit_idx])
        z    = 1 - 2 * bit  # 0->+1, 1->-1
        expval += z * count / total
    return expval


def statevector_expvals(x_in, feedback, n_qubits, h_field, J):
    """Reference statevector simulation for comparison."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, SparsePauliOp
        import qiskit.quantum_info as qi

        qc = build_tfim_circuit(x_in, feedback, n_qubits, h_field, J)
        # Remove measurements for statevector
        qc_sv = qc.remove_final_measurements(inplace=False)
        sv = Statevector(qc_sv)
        evs = []
        for i in range(n_qubits):
            zop = SparsePauliOp("I" * (n_qubits-1-i) + "Z" + "I" * i)
            evs.append(float(sv.expectation_value(zop).real))
        return np.array(evs)
    except Exception as e:
        print(f"Statevector failed: {e}")
        return np.zeros(n_qubits)


def run_on_ibm(circuits, shots, token):
    """Submit circuits to IBM Quantum and retrieve results."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(channel="ibm_quantum", token=token)

    # Find least-busy backend with enough qubits
    backends = service.backends(
        filters=lambda b: (b.configuration().n_qubits >= N_QUBITS
                           and not b.configuration().simulator
                           and b.status().operational)
    )
    backend = min(backends, key=lambda b: b.status().pending_jobs)
    print(f"  Selected backend: {backend.name} "
          f"({backend.configuration().n_qubits} qubits, "
          f"{backend.status().pending_jobs} pending jobs)")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = [pm.run(qc) for qc in circuits]

    sampler = SamplerV2(backend)
    job = sampler.run(transpiled, shots=shots)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results...")

    result = job.result()
    counts_list = []
    for i in range(len(circuits)):
        pub_result = result[i]
        counts = pub_result.data.c.get_counts()
        counts_list.append(counts)

    return counts_list, backend.name


def main():
    print("=" * 65)
    print("HF-QRC QPU Validation — Qudit Creons | GIC 2026")
    print(f"N={N_QUBITS} qubits, T={T_STEPS} step, {SHOTS} shots/circuit")
    print("=" * 65)

    # Check token
    if IBM_TOKEN == "PASTE_YOUR_IBM_TOKEN_HERE":
        print("\nERROR: Set IBM_TOKEN environment variable:")
        print("  export IBM_TOKEN='your_token_from_quantum.ibm.com'")
        print("\nAlternatively, paste your token in qpu_run.py line 51.")
        print("\nFor now, running statevector simulation only as demo...")
        run_statevector_demo()
        return

    # Fixed coupling matrix (same seed as pipeline)
    rng = np.random.default_rng(SEED)
    J = rng.standard_normal((N_QUBITS, N_QUBITS))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    # Generate test inputs (subset of log-RV values)
    rng2   = np.random.default_rng(SEED + 1)
    inputs = rng2.standard_normal(N_SAMPLES) * 0.3 - 4.0

    # Build circuits
    print(f"\nBuilding {N_SAMPLES} QPU circuits...")
    circuits = []
    sv_expvals = []
    for i, x in enumerate(inputs):
        qc = build_tfim_circuit(x, 0.0, N_QUBITS, H_FIELD, J)
        circuits.append(qc)
        sv_ev = statevector_expvals(x, 0.0, N_QUBITS, H_FIELD, J)
        sv_expvals.append(sv_ev)
        if (i+1) % 5 == 0:
            print(f"  Built {i+1}/{N_SAMPLES}")

    sv_expvals = np.array(sv_expvals)

    # Run on IBM QPU
    print(f"\nSubmitting {N_SAMPLES} circuits to IBM Quantum...")
    t0 = time.time()
    try:
        counts_list, backend_name = run_on_ibm(circuits, SHOTS, IBM_TOKEN)
        elapsed = time.time() - t0

        # Extract shot-based expectation values
        shot_expvals = np.array([
            [shots_to_expval(counts, i, N_QUBITS) for i in range(N_QUBITS)]
            for counts in counts_list
        ])

        # Compare hardware vs. statevector
        diff = np.abs(shot_expvals - sv_expvals)
        mean_diff = float(np.mean(diff))
        max_diff  = float(np.max(diff))

        print(f"\nHardware vs. Statevector comparison:")
        print(f"  Mean |Δ<Z_i>|: {mean_diff:.4f}")
        print(f"  Max  |Δ<Z_i>|: {max_diff:.4f}")
        print(f"  Wall-clock: {elapsed:.1f}s")
        print(f"  Backend: {backend_name}")
        print(f"  Shots: {SHOTS} per circuit")
        print(f"  Circuits run: {N_SAMPLES}")

        # Save results
        qpu_out = {
            "backend": backend_name,
            "n_qubits": N_QUBITS,
            "shots": SHOTS,
            "n_circuits": N_SAMPLES,
            "wall_clock_s": round(elapsed, 1),
            "mean_abs_diff_expval": round(mean_diff, 4),
            "max_abs_diff_expval":  round(max_diff, 4),
            "circuit_depth_before_transpile": 3 * N_QUBITS,
            "hardware_expvals": shot_expvals.tolist(),
            "statevector_expvals": sv_expvals.tolist(),
        }

        summary = [
            "=" * 55,
            "QPU VALIDATION SUMMARY",
            f"  Backend:            {backend_name}",
            f"  Qubits used:        {N_QUBITS}",
            f"  Shots per circuit:  {SHOTS}",
            f"  Circuits run:       {N_SAMPLES}",
            f"  Circuit depth:      ~{3*N_QUBITS} gates (pre-transpile)",
            f"  Wall-clock time:    {elapsed:.1f}s",
            "",
            "  Expectation value fidelity (hardware vs. statevector):",
            f"  Mean |Δ<Z_i>|: {mean_diff:.4f}",
            f"  Max  |Δ<Z_i>|: {max_diff:.4f}",
            "",
            "  Interpretation:",
            f"  Hardware noise causes ~{mean_diff*100:.1f}% average deviation",
            "  from noiseless simulation per observable.",
            "=" * 55,
        ]
        summary_str = "\n".join(summary)
        print("\n" + summary_str)

    except Exception as e:
        print(f"\nQPU run failed: {e}")
        print("Falling back to statevector demo...")
        qpu_out = {"error": str(e), "statevector_expvals": sv_expvals.tolist()}
        summary_str = f"QPU run failed: {e}\nStatevector demo run instead."

    with open("results/qpu_results.json", "w") as f:
        json.dump(qpu_out, f, indent=2)
    with open("results/qpu_summary.txt", "w") as f:
        f.write(summary_str)
    print("\nSaved: results/qpu_results.json")
    print("Saved: results/qpu_summary.txt")


def run_statevector_demo():
    """Demo mode: just run statevector simulation, no QPU needed."""
    print("\n[DEMO MODE] Running statevector simulation (no QPU token)")
    rng = np.random.default_rng(SEED)
    J = rng.standard_normal((N_QUBITS, N_QUBITS))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    rng2   = np.random.default_rng(SEED + 1)
    inputs = rng2.standard_normal(N_SAMPLES) * 0.3 - 4.0

    print(f"Computing {N_SAMPLES} statevector expectation values...")
    evs = []
    for x in inputs:
        ev = statevector_expvals(x, 0.0, N_QUBITS, H_FIELD, J)
        evs.append(ev)

    evs = np.array(evs)
    print(f"Mean <Z_i> across samples: {evs.mean():.4f}")
    print(f"Std  <Z_i> across samples: {evs.std():.4f}")
    print("Demo complete. Set IBM_TOKEN to run on real QPU.")

    out = {"mode": "statevector_demo", "statevector_expvals": evs.tolist()}
    with open("results/qpu_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/qpu_results.json")


if __name__ == "__main__":
    main()
