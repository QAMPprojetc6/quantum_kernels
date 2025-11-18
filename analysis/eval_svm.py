"""
Evaluate SVM with a precomputed kernel.

CLI:
    python analysis/eval_svm.py \
      --kernel outputs/K_local-make_circles_42.npy \
      --splits outputs/splits_make_circles_42.json \
      --C 0.1 1 10 \
      --out outputs/metrics.csv
"""

import argparse
import json
import os
from typing import List

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv


def read_splits(path: str):
    """
    Expected JSON with keys: train_idx, val_idx, test_idx, y_all
    All indices refer to positions in the full dataset used to build K.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return (
        np.array(data["train_idx"], dtype=int),
        np.array(data["val_idx"], dtype=int),
        np.array(data["test_idx"], dtype=int),
        np.array(data["y_all"]),
    )


def eval_precomputed(K: np.ndarray, y: np.ndarray, train_idx, val_idx, test_idx, Cs: List[float]):
    # Centering and scaling of K can be added here if needed.
    best_val_acc = -1.0
    best_C = None
    best_model = None

    for C in Cs:
        clf = SVC(C=C, kernel="precomputed")
        K_train = K[np.ix_(train_idx, train_idx)]
        clf.fit(K_train, y[train_idx])

        K_val = K[np.ix_(val_idx, train_idx)]
        y_val_pred = clf.predict(K_val)
        val_acc = accuracy_score(y[val_idx], y_val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
            best_model = clf

    # Test with best C
    K_test = K[np.ix_(test_idx, train_idx)]
    y_test_pred = best_model.predict(K_test)
    test_acc = accuracy_score(y[test_idx], y_test_pred)

    return {
        "best_C": best_C,
        "val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
    }


def main(kernel_path: str, splits_path: str, Cs: List[float], out_csv: str):
    K = np.load(kernel_path)
    train_idx, val_idx, test_idx, y_all = read_splits(splits_path)
    metrics = eval_precomputed(K, y_all, train_idx, val_idx, test_idx, Cs)

    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    header = ["kernel_path", "splits_path", "best_C", "val_acc", "test_acc"]
    row = [kernel_path, splits_path, metrics["best_C"], metrics["val_acc"], metrics["test_acc"]]

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[OK] Wrote metrics to {out_csv}: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SVM with precomputed kernels")
    parser.add_argument("--kernel", required=True, help="Path to K.npy")
    parser.add_argument("--splits", required=True, help="Path to splits_*.json")
    parser.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="List of C values")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    main(args.kernel, args.splits, args.C, args.out)
