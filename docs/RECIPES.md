# Multiscale Kernel: Ready-to-Run Recipes (one-liners)

> Run from the **repo root**. Outputs go to `outputs/multiscale/` and figures to `figs/multiscale/` (as configured in the script).

We keep **consistent case names** across datasets:

* **Baseline (Global-only)** — compare states on **all qubits only**. (No locality; strongest chance to suffer concentration.)
* **Local-only (1q patches)** — compare states on **single-qubit patches only**. (Purely local signal.)
* **Multi-Scale (Local+Global)** — mix **1q local** and **global** scales with non-negative weights. (Intended to preserve local structure while keeping global context.)

> Why 1-qubit patches for “Local”? Because it works for both `make_circles` (d=2) and `iris` (d=4), giving parity across datasets.
> (For Iris you *can* also try pairs `(0,1),(2,3)`; see the “Optional variants” section at the end.)

---

## Dataset: `make_circles` (d = 2)

### Baseline (Global-only)
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0,1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_global
```

### Local-only (1q patches)
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_local1q
```

### Multi-Scale (Local+Global)
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]], [[0,1]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_circles_ms_local1q_global
```

*(Optional for analysis)* add `--center --report-rank` to any command.

## Dataset: `iris` (d = 4)

### Baseline (Global-only)
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1,2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_global
```

### Local-only (1q patches)
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0],[1],[2],[3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_local1q
```

### Multi-Scale (Local+Global)
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0],[1],[2],[3]], [[0,1,2,3]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_iris_ms_local1q_global
```

*(Optional for analysis)* add `--center --report-rank` to any command.

---

## What each scenario is testing

* **Baseline (Global-only)**
  Uses **one patch: all qubits**. This is the standard global kernel (no locality). Good to expose **exponential concentration** risks.

* **Local-only (1q patches)**
  Uses **only single-qubit RDMs** (e.g., for d=4: patches `[0],[1],[2],[3]`). Tests whether purely local structure carries discriminative signal.

* **Multi-Scale (Local+Global)**
  Weighted sum of **local 1q** and **global** kernels (here 50/50). Goal: **retain** local contrast while keeping **global** context—often more robust than either alone.

---

## Optional variants (Iris, pairs instead of 1q)

If you want a second kind of “local” on Iris (d=4), try **pairs** `(0,1)` and `(2,3)`:

* Local-only (pairs):
  `--scales '[[[0,1],[2,3]]]' --weights '[1.0]'`
* Multi-Scale (pairs + all):
  `--scales '[[[0,1],[2,3]], [[0,1,2,3]]]' --weights '[0.6, 0.4]'`

---

## Notes

* Add `--center` to remove the mean component (useful for spectrum analysis).
* Add `--report-rank` to print & save effective rank and eigenvalue stats.
* JSON quoting: **PowerShell** accepts `'...'` inside the command; **CMD** needs `"..."`; **Bash/zsh** prefers `'...'`.