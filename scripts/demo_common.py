"""
Shared helpers for demo scripts.

This module factors out the duplicated logic between:
  - scripts/run_multiscale_demo.py
  - scripts/run_baseline_demo.py

Important:
- Keep behavior identical to the original scripts: same preprocessing, splits,
  centering, spectrum reporting, artifact naming, plotting, and CSV schema.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


# ---------------------------
# data helpers
# ---------------------------
def load_dataset(
    name: str,
    n_samples: int,
    seed: int,
    n_features: Optional[int] = None,
    pca: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a small dataset.

    Supported:
      - make_circles (synthetic, uses n_samples)
      - iris (full dataset)
      - breast_cancer (full dataset)
      - parkinsons (OpenML id 1488; downloads on first use)
      - star_classification (local CSV; filters to classes GALAXY and STAR)
        If n_features exceeds the raw columns, we add simple pairwise interactions.
      - exam_score_prediction (local CSV; pass/fail from exam_score >= 60)
        If n_features exceeds the raw columns, we add simple pairwise interactions.
      - ionosphere (local CSV; labels in last column, 'g' or 'b')
      - heart_disease (local CSV; target in column 'num', binary)
        If n_features exceeds the raw columns, we add simple pairwise interactions.

    Preprocessing:
      - shuffle with RNG(seed)
      - StandardScaler
      - optional dimensionality reduction to n_features (PCA or simple truncation)
      - map to radians: X <- pi * X / 2
    """
    from sklearn.datasets import make_circles, load_iris, load_breast_cancer, fetch_openml
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)

    if name == "make_circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.1, random_state=seed)

    elif name == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target

    elif name in {"breast_cancer", "breast-cancer", "cancer"}:
        bc = load_breast_cancer()
        X, y = bc.data, bc.target  # binary

    elif name in {"parkinsons", "parkinson", "parkinson_disease", "parkinson-disease"}:
        # OpenML Parkinsons dataset (UCI voice measures), target is "status" (0/1)
        ds = fetch_openml(data_id=1488, as_frame=False)  # downloads/caches on first call
        X, y = ds.data, ds.target

    elif name in {"star_classification", "star-classification", "star", "stars"}:
        # SDSS star/galaxy classification (CSV in project); filter to GALAXY and STAR.
        csv_path = Path(__file__).resolve().parents[1] / "datasets" / "star_classification.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        keep_labels = {"GALAXY", "STAR"}
        X_rows = []
        y_rows = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "class" not in reader.fieldnames:
                raise ValueError("star_classification.csv must include a 'class' column.")
            feature_cols = [c for c in reader.fieldnames if c != "class"]

            for row in reader:
                label = (row.get("class") or "").strip().upper()
                if label not in keep_labels:
                    continue
                try:
                    feats = [float(row[c]) for c in feature_cols]
                except (KeyError, TypeError, ValueError):
                    continue
                X_rows.append(feats)
                y_rows.append(label)

        if not X_rows:
            raise ValueError("No rows found for classes GALAXY/STAR in star_classification.csv.")

        X = np.asarray(X_rows, dtype=np.float64)
        y = np.array([0 if lbl == "GALAXY" else 1 for lbl in y_rows], dtype=int)

        if n_samples is not None and int(n_samples) > 0 and X.shape[0] > int(n_samples):
            pick = rng.choice(X.shape[0], size=int(n_samples), replace=False)
            X = X[pick]
            y = y[pick]

        if n_features is not None and int(n_features) > X.shape[1]:
            target = int(n_features)
            base_d = X.shape[1]
            need = target - base_d
            pairs = []
            for i in range(base_d):
                for j in range(i + 1, base_d):
                    pairs.append((i, j))
                    if len(pairs) >= need:
                        break
                if len(pairs) >= need:
                    break
            if len(pairs) < need:
                for i in range(base_d):
                    pairs.append((i, i))
                    if len(pairs) >= need:
                        break
            if len(pairs) < need:
                raise ValueError(
                    "Not enough interaction features available to reach n_features."
                )
            if pairs:
                inter = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
                X = np.concatenate([X, inter], axis=1)

    elif name in {"exam_score_prediction", "exam-score-prediction", "exam_score", "exam"}:
        # Exam score prediction -> binary classification (pass >= 60).
        csv_path = Path(__file__).resolve().parents[1] / "datasets" / "Exam_Score_Prediction.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        def _is_float(val: str) -> bool:
            try:
                float(val)
                return True
            except (TypeError, ValueError):
                return False

        rows = []
        y_rows = []
        fieldnames = None
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames is None or "exam_score" not in fieldnames:
                raise ValueError("Exam_Score_Prediction.csv must include an 'exam_score' column.")

            for row in reader:
                score_raw = row.get("exam_score", "")
                if not _is_float(score_raw):
                    continue
                score = float(score_raw)
                y_rows.append(1 if score >= 60.0 else 0)
                rows.append(row)

        if not rows:
            raise ValueError("No valid rows found in Exam_Score_Prediction.csv.")

        drop_cols = {"exam_score", "student_id"}
        feature_cols = [c for c in fieldnames if c not in drop_cols]

        numeric_cols = []
        categorical_cols = []
        for col in feature_cols:
            vals = [r.get(col, "") for r in rows]
            if all((v == "" or _is_float(v)) for v in vals):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        if numeric_cols:
            X_num = np.array(
                [[float(r.get(c, 0.0) or 0.0) for c in numeric_cols] for r in rows],
                dtype=np.float64,
            )
        else:
            X_num = np.zeros((len(rows), 0), dtype=np.float64)

        if categorical_cols:
            X_cat = [[(r.get(c, "") or "") for c in categorical_cols] for r in rows]
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat_enc = enc.fit_transform(X_cat)
            X = np.concatenate([X_num, X_cat_enc], axis=1)
        else:
            X = X_num

        y = np.array(y_rows, dtype=int)

        if n_samples is not None and int(n_samples) > 0 and X.shape[0] > int(n_samples):
            pick = rng.choice(X.shape[0], size=int(n_samples), replace=False)
            X = X[pick]
            y = y[pick]

        if n_features is not None and int(n_features) > X.shape[1]:
            target = int(n_features)
            base_d = X.shape[1]
            need = target - base_d
            pairs = []
            for i in range(base_d):
                for j in range(i + 1, base_d):
                    pairs.append((i, j))
                    if len(pairs) >= need:
                        break
                if len(pairs) >= need:
                    break
            if len(pairs) < need:
                for i in range(base_d):
                    pairs.append((i, i))
                    if len(pairs) >= need:
                        break
            if len(pairs) < need:
                raise ValueError(
                    "Not enough interaction features available to reach n_features."
                )
            if pairs:
                inter = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
                X = np.concatenate([X, inter], axis=1)

    elif name in {"ionosphere", "iono"}:
        # Ionosphere dataset: 34 continuous features + class label in last column ('g'/'b').
        csv_path = Path(__file__).resolve().parents[1] / "datasets" / "ionosphere.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        X_rows = []
        y_rows = []
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                label = (row[-1] or "").strip().lower()
                if label not in {"g", "b"}:
                    continue
                try:
                    feats = [float(v) for v in row[:-1]]
                except (TypeError, ValueError):
                    continue
                X_rows.append(feats)
                y_rows.append(1 if label == "g" else 0)

        if not X_rows:
            raise ValueError("No valid rows found in ionosphere.csv.")

        X = np.asarray(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=int)

        if n_samples is not None and int(n_samples) > 0 and X.shape[0] > int(n_samples):
            pick = rng.choice(X.shape[0], size=int(n_samples), replace=False)
            X = X[pick]
            y = y[pick]

    elif name in {"heart_disease", "heart-disease", "heart"}:
        # Heart disease dataset: target is column 'num' (0 = no disease, >0 = disease).
        csv_path = Path(__file__).resolve().parents[1] / "datasets" / "heart_disease_all_sites.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        def _is_float(val: str) -> bool:
            try:
                float(val)
                return True
            except (TypeError, ValueError):
                return False

        rows = []
        y_rows = []
        fieldnames = None
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames is None or "num" not in fieldnames:
                raise ValueError("heart_disease_all_sites.csv must include a 'num' column.")

            for row in reader:
                num_raw = row.get("num", "")
                if not _is_float(num_raw):
                    continue
                num = float(num_raw)
                y_rows.append(1 if num > 0 else 0)
                rows.append(row)

        if not rows:
            raise ValueError("No valid rows found in heart_disease_all_sites.csv.")

        drop_cols = {"num", "site"}
        feature_cols = [c for c in fieldnames if c not in drop_cols]

        numeric_cols = []
        categorical_cols = []
        for col in feature_cols:
            vals = [r.get(col, "") for r in rows]
            if all((v == "" or _is_float(v)) for v in vals):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        if numeric_cols:
            X_num = np.array(
                [[float(r.get(c, 0.0) or 0.0) for c in numeric_cols] for r in rows],
                dtype=np.float64,
            )
        else:
            X_num = np.zeros((len(rows), 0), dtype=np.float64)

        if categorical_cols:
            X_cat = [[(r.get(c, "") or "") for c in categorical_cols] for r in rows]
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat_enc = enc.fit_transform(X_cat)
            X = np.concatenate([X_num, X_cat_enc], axis=1)
        else:
            X = X_num

        y = np.array(y_rows, dtype=int)

        if n_samples is not None and int(n_samples) > 0 and X.shape[0] > int(n_samples):
            pick = rng.choice(X.shape[0], size=int(n_samples), replace=False)
            X = X[pick]
            y = y[pick]

        if n_features is not None and int(n_features) > X.shape[1]:
            target = int(n_features)
            base_d = X.shape[1]
            need = target - base_d
            pairs = []
            for i in range(base_d):
                for j in range(i + 1, base_d):
                    pairs.append((i, j))
                    if len(pairs) >= need:
                        break
                if len(pairs) >= need:
                    break
            if len(pairs) < need:
                for i in range(base_d):
                    pairs.append((i, i))
                    if len(pairs) >= need:
                        break
            if len(pairs) < need:
                raise ValueError(
                    "Not enough interaction features available to reach n_features."
                )
            if pairs:
                inter = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
                X = np.concatenate([X, inter], axis=1)

    else:
        raise ValueError(
            "dataset must be one of: make_circles, iris, breast_cancer, parkinsons, "
            "star_classification, exam_score_prediction, ionosphere, heart_disease."
        )

    # Shuffle (iris comes ordered)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X = X.astype(np.float64)

    # standardize
    X = StandardScaler().fit_transform(X)

    # optional dimensionality reduction
    if n_features is not None and int(n_features) > 0 and X.shape[1] > int(n_features):
        k = int(n_features)
        if pca:
            X = PCA(n_components=k, random_state=seed).fit_transform(X)
            # re-standardize after PCA (keeps angle scaling more consistent)
            X = StandardScaler().fit_transform(X)
        else:
            X = X[:, :k]

    # map to angles
    X = np.pi * X / 2.0

    # OpenML can return y as strings; make it int if possible
    try:
        y = y.astype(int)
    except Exception:
        y = np.array(y, dtype=int)

    return X, y


def make_splits(n: int, seed: int, val_size: float, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic train/val/test splits.

    Behavior matches the current demo scripts:
      - no stratify
      - use val_size + test_size as holdout, then split holdout into val/test.
    """
    from sklearn.model_selection import train_test_split

    idx_all = np.arange(n, dtype=int)
    idx_train, idx_tmp = train_test_split(
        idx_all,
        test_size=(val_size + test_size),
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=rel_test,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


# ---------------------------
# kernel post-processing
# ---------------------------
def center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center the kernel: Kc = H K H. Enforces exact symmetry."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    return 0.5 * (Kc + Kc.T)


def spectrum_stats(K: np.ndarray, thresh: float = 1e-6) -> Dict:
    """Return basic spectrum stats for symmetric K."""
    Ks = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ks)
    eff_rank = int(np.sum(w > thresh))
    return {
        "n": int(K.shape[0]),
        "trace": float(np.sum(w)),
        "lambda_min": float(w.min()),
        "lambda_max": float(w.max()),
        "effective_rank@{:.0e}".format(thresh): eff_rank,
        "threshold": float(thresh),
    }


def normalize_unit_diag(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize kernel to unit diagonal: K_ij <- K_ij / sqrt(K_ii K_jj)."""
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    Kn = K / (d[:, None] * d[None, :])
    Kn = 0.5 * (Kn + Kn.T)
    np.fill_diagonal(Kn, 1.0)
    return Kn.astype(np.float64)


# ---------------------------
# I/O helpers
# ---------------------------
def prepare_dirs(out_prefix: Path, figs_subdir: str) -> Tuple[Path, Path]:
    """
    Create output dir (parent of out_prefix) and figures dir.
    Returns: (out_dir, figs_dir)
    """
    out_dir = out_prefix.parent
    figs_dir = Path(figs_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, figs_dir


def artifact_paths(out_prefix: Path, figs_dir: Path, centered: bool) -> Dict[str, str]:
    """
    Match the current demo naming scheme:
      suffix = "_centered" if centered else ""
      kpath = f"{out_prefix}_K{suffix}.npy"
      mpath = f"{out_prefix}_meta.json"
      spath = f"{out_prefix}_splits.json"
      fprefix = f"{figs_dir}/{out_prefix.name}{suffix}"
    """
    suffix = "_centered" if centered else ""
    return {
        "suffix": suffix,
        "kpath": str(out_prefix) + f"_K{suffix}.npy",
        "mpath": str(out_prefix) + "_meta.json",
        "spath": str(out_prefix) + "_splits.json",
        "fprefix": str(figs_dir / (out_prefix.name + suffix)),
        "spectrum_path": str(out_prefix) + "_spectrum.txt",
    }


def save_json(path: str, obj: Dict) -> None:
    def _default(o):
        # NumPy scalars -> Python scalars
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        # NumPy arrays -> lists
        if isinstance(o, np.ndarray):
            return o.tolist()
        # Fallback
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_splits(path: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, y_all: np.ndarray) -> None:
    save_json(
        path,
        {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "y_all": y_all.tolist(),
        },
    )


def write_spectrum_txt(path: str, stats: Dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def append_metrics_row(metrics_csv: str, header: list, row: list) -> None:
    """
    Append a row with the same CSV schema used by the demo scripts.
    """
    write_header = not os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if write_header:
            w.writerow(header)
        w.writerow(row)


def plot_all(K: np.ndarray, fprefix: str) -> None:
    """
    Generate the three standard diagnostic figures.
    """
    from analysis.diagnostics import plot_heatmap, plot_offdiag_hist, plot_spectrum

    plot_heatmap(K, f"{fprefix}_matrix.png")
    plot_offdiag_hist(K, f"{fprefix}_offdiag_hist.png")
    plot_spectrum(K, f"{fprefix}_spectrum.png")
