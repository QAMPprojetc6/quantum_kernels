from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# --- 1) Loader for the multi-site "processed.*.data" files ---

PROCESSED_COLS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","num"
]

SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

def load_processed_multisite(root_dir: str | Path, include_sites=None) -> pd.DataFrame:
    """
    Load and merge the multi-site UCI Heart Disease processed datasets.

    Parameters
    ----------
    root_dir : path to the extracted zip folder (the one containing processed.*.data)
    include_sites : iterable of site keys to include (default: all SITE_FILES)

    Returns
    -------
    df : merged DataFrame with a 'site' column
    """
    root = Path(root_dir)
    if include_sites is None:
        include_sites = list(SITE_FILES.keys())

    frames = []
    for site in include_sites:
        fname = SITE_FILES[site]
        fpath = root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing file: {fpath}")

        # These files are CSV-like, no header, missing values are usually '?'
        df_site = pd.read_csv(
            fpath,
            header=None,
            names=PROCESSED_COLS,
            na_values=["?"],
            skipinitialspace=True
        )
        df_site["site"] = site
        frames.append(df_site)

    df = pd.concat(frames, ignore_index=True)

    # Convert everything except 'site' to numeric (some columns may be read as object due to '?')
    for c in PROCESSED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --- 2) Target setup helpers ---

def make_binary_target(df: pd.DataFrame) -> pd.Series:
    """
    UCI target 'num' is typically 0..4 (0 = no disease, 1-4 = disease severity).
    This makes a common binary target: 0 vs >0
    """
    return (df["num"] > 0).astype(int)

def make_multiclass_target(df: pd.DataFrame) -> pd.Series:
    """Keep the original 0..4 labels (drops NaNs)."""
    return df["num"].astype("Int64")  # nullable int


# --- 3) Preprocessing + model pipeline ---

def build_pipeline(X: pd.DataFrame):
    """
    Creates a preprocessing pipeline:
    - numeric: median impute + scale
    - categorical-like coded ints: most-frequent impute + one-hot
    - includes 'site' as categorical
    """
    # In this dataset, many integer-coded columns are categorical:
    categorical = ["sex","cp","fbs","restecg","exang","slope","ca","thal","site"]
    numeric = ["age","trestbps","chol","thalach","oldpeak"]

    # Some files may have all columns numeric but still categorical in meaning.
    # We'll enforce those lists if present in X.
    categorical = [c for c in categorical if c in X.columns]
    numeric = [c for c in numeric if c in X.columns]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", categorical_tf, categorical),
        ],
        remainder="drop"
    )

    # Baseline model (change to RandomForest/XGBoost/etc. if you want)
    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe


# --- 4) End-to-end example (binary classification) ---

if __name__ == "__main__":
    # Point this at the folder where your files are extracted:
    # e.g. "/path/to/heart-disease/"
    DATA_DIR = "./all_heart_disease_sites/"

    df = load_processed_multisite(DATA_DIR)

    # write merged multi-site dataset to a single file (with headers)
    out_path = Path(DATA_DIR) / "heart_disease_processed_all_sites.csv"
    df["num"] = (df["num"] > 0).astype(int)  # make target binary (0=no disease, 1=disease)
    df.to_csv(out_path, index=False, line_terminator="\n", encoding="utf-8")  # includes headers by default
    print(f"Saved merged dataset to: {out_path.resolve()}")

    # Drop rows with missing target
    df = df.dropna(subset=["num"]).copy()

    # Choose target style:
    y = make_binary_target(df)        # binary
    # y = make_multiclass_target(df)  # multiclass (0..4)

    X = df.drop(columns=["num"])

    # Train/test split (stratify on target; you can also stratify on site if you want)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    # For binary: AUC
    if len(np.unique(y_test)) == 2:
        proba = pipe.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, proba))
