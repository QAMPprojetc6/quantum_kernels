# Multiscale Kernel: Ready-to-Run Recipes (one-liners)

> Run from the **repo root**. Outputs go to `outputs/multiscale/` and figures to `figs/multiscale/` (as configured in the script).

We keep **consistent case names** across datasets:

* **Baseline (all-qubits only)** — compare states on **all qubits only**. (No locality; strongest chance to suffer concentration.)
* **Local-only (1q patches)** — compare states on **single-qubit patches only**. (Purely local signal.)
* **Multi-Scale (Local + Baseline)** — mix **local** and **all-qubits** scales with non-negative weights. (Preserve local structure while keeping global context.)
* **Multi-Scale default (pairs + all)** — the script default when you omit `--scales/--weights`.


> Why 1-qubit patches for “Local”? Because it works for both `make_circles` (d=2) and `iris` (d=4), giving parity across datasets.

> For Iris you can also try 2-qubit patches; see the “Optional variants” section.

---

## Dataset: `make_circles` (d = 2)

### Baseline (all-qubits only)
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0,1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_baseline
```

### Local-only (1q patches)
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_local1q
```

### Multi-Scale (Local 1q + Baseline all-qubits)
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]], [[0,1]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_circles_ms_local1q_baseline
```

### Multi-Scale default (pairs + all) — no `--scales/--weights`
```
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --out-prefix outputs/multiscale/ms_circles_default_pairs_all
```

> Note: for `make_circles` (d=2), "pairs" and "all" are the same patch `[0,1]`,
> so the default effectively repeats the same scale twice with uniform weights.

*(Optional for analysis)* add `--center --report-rank` to any command.


## Dataset: `iris` (d = 4)

### Baseline (all-qubits only)
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1,2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_baseline
```

### Local-only (1q patches)
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0],[1],[2],[3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_local1q
```

### Multi-Scale (Local 1q + Baseline all-qubits)
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0],[1],[2],[3]], [[0,1,2,3]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_iris_ms_local1q_baseline
```

### Multi-Scale default (pairs + all) — no `--scales/--weights`
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/multiscale/ms_iris_default_pairs_all
```

*(Optional for analysis)* add `--center --report-rank` to any command.

---

## What each scenario is testing

* **Baseline (all-qubits only)**  
  Uses **one patch: all qubits**. This is the standard baseline fidelity-style comparison (no locality). Good to expose **exponential concentration** risks.

* **Local-only (1q patches)**  
  Uses **only single-qubit patches**. Tests whether purely local structure retains informative variance.

* **Multi-Scale (Local + Baseline)**  
  Weighted sum of a local scale and the all-qubits scale. Goal: preserve local contrast while retaining global context.

* **Default (pairs + all)**  
  Uses pairs at the first scale and all-qubits at the second scale, with uniform weights.

---

## Optional variants (Iris, 2q patches)

If you want a second kind of “local” on Iris (d=4), try **pairs** `(0,1)` and `(2,3)`:

### Local-only (2q patches):
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1],[2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_local2q
```

### Multi-Scale (Local 2q + Baseline all-qubits):
```
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1],[2,3]], [[0,1,2,3]]]' --weights '[0.6, 0.4]' --out-prefix outputs/multiscale/ms_iris_ms_local2q_baseline --center --report-rank
```

---

## Notes

* Add `--center` to remove the mean component (useful for spectrum analysis).
* Add `--report-rank` to print & save effective rank and eigenvalue stats.
* Optional normalization flags:
  * `--normalize` / `--no-normalize`
* JSON quoting: **PowerShell** accepts `'...'` inside the command; **CMD** needs `"..."`; **Bash/zsh** prefers `'...'`.