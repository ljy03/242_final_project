"""
Person 3 — Classical ML Modeler (TF-IDF Models) — IVO

Inputs (from Person 1):
- outputs/tfidf_features.pkl  (X_train, X_val, X_test, optionally y_train/y_val/y_test and vectorizer)
- outputs/clean_reviews.csv   (fallback for labels if y_* not present in pickle)

Outputs (generated locally; should be gitignored):
- outputs/tfidf_model_comparison.csv
- Console: metrics table, confusion matrices, best model choice

Models required:
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

try:
    import pandas as pd
except ImportError:
    pd = None  # will error later if needed

# XGBoost is optional dependency; required by your spec, but we handle gracefully if missing
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False


RANDOM_STATE = 42


@dataclass
class EvalResult:
    model_name: str
    split: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_time_sec: float
    conf_mat: np.ndarray


def _ensure_outputs_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _binary_labels_from_sentiment(series) -> np.ndarray:
    """
    Accepts labels in various formats:
    - 'positive'/'negative'
    - 1/0
    - True/False
    """
    if series.dtype == object:
        s = series.astype(str).str.lower()
        return np.where(s.isin(["positive", "pos", "1", "true", "t", "yes"]), 1, 0).astype(int)
    return series.astype(int).to_numpy()


def load_tfidf_bundle(
    tfidf_pkl_path: str,
    clean_csv_path: Optional[str] = None,
    seed: int = RANDOM_STATE,
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads X_train/X_val/X_test from pickle.
    Tries to load y_train/y_val/y_test from pickle; if missing, reconstruct labels from clean_reviews.csv.

    This keeps Person 3 independent while still robust if Person 1 didn't store y_*.
    """
    with open(tfidf_pkl_path, "rb") as f:
        bundle = pickle.load(f)

    # Accept either dict-like or object with attributes
    def get_key(k: str):
        if isinstance(bundle, dict):
            return bundle.get(k, None)
        return getattr(bundle, k, None)

    X_train = get_key("X_train")
    X_val = get_key("X_val")
    X_test = get_key("X_test")

    if X_train is None or X_val is None or X_test is None:
        raise ValueError(
            "tfidf_features.pkl is missing one of X_train/X_val/X_test. "
            "Expected keys: X_train, X_val, X_test."
        )

    y_train = get_key("y_train")
    y_val = get_key("y_val")
    y_test = get_key("y_test")

    if y_train is not None and y_val is not None and y_test is not None:
        return X_train, X_val, X_test, np.asarray(y_train), np.asarray(y_val), np.asarray(y_test)

    # Fallback: derive labels and reproduce split deterministically
    if clean_csv_path is None:
        raise ValueError(
            "y_train/y_val/y_test not found in tfidf_features.pkl and no clean_reviews.csv provided."
        )
    if pd is None:
        raise ImportError("pandas is required to load clean_reviews.csv for label fallback.")

    df = pd.read_csv(clean_csv_path)

    # Try common label column names
    label_col = None
    for cand in ["sentiment", "label", "target", "y"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(
            "Could not find label column in clean_reviews.csv. "
            "Expected one of: sentiment, label, target, y"
        )

    y_all = _binary_labels_from_sentiment(df[label_col])

    # Recreate the same 70/10/20 split as README: train/val/test = 70/10/20 stratified
    idx = np.arange(len(y_all))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y_all, test_size=0.20, random_state=seed, stratify=y_all
    )
    # val is 10% of total => 10/80 = 12.5% of trainval
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval, y_trainval, test_size=0.125, random_state=seed, stratify=y_trainval
    )

    # IMPORTANT: This assumes X_train/X_val/X_test correspond to that same deterministic split.
    # If Person 1 used different seed, you'll want y_* stored in the pickle.
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(
    model,
    X,
    y: np.ndarray,
    split_name: str,
    train_time_sec: float,
    model_name: str,
) -> EvalResult:
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, y_pred)

    return EvalResult(
        model_name=model_name,
        split=split_name,
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        train_time_sec=float(train_time_sec),
        conf_mat=cm,
    )


def fit_with_val_tuning(
    model_ctor,
    param_grid: Dict[str, List[Any]],
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    model_name: str,
) -> Tuple[Any, Dict[str, Any], float, EvalResult]:
    """
    Light, explainable tuning: iterate a small grid, choose best by validation F1.
    Returns best_model, best_params, best_train_time, best_val_result
    """
    best = None
    best_params: Dict[str, Any] = {}
    best_val: Optional[EvalResult] = None
    best_time = float("inf")

    # Make list of param dicts (no sklearn dependency on ParameterGrid to keep it simple)
    keys = list(param_grid.keys())
    values_list = [param_grid[k] for k in keys]

    def rec_build(i: int, cur: Dict[str, Any], out: List[Dict[str, Any]]):
        if i == len(keys):
            out.append(dict(cur))
            return
        for v in values_list[i]:
            cur[keys[i]] = v
            rec_build(i + 1, cur, out)

    combos: List[Dict[str, Any]] = []
    rec_build(0, {}, combos)

    for params in combos:
        model = model_ctor(**params)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        train_time = t1 - t0

        val_res = evaluate(model, X_val, y_val, "val", train_time, model_name)
        if best_val is None or (val_res.f1 > best_val.f1) or (
            np.isclose(val_res.f1, best_val.f1) and val_res.accuracy > best_val.accuracy
        ):
            best = model
            best_params = params
            best_val = val_res
            best_time = train_time

    assert best is not None and best_val is not None
    return best, best_params, best_time, best_val


def print_result(res: EvalResult) -> None:
    print(
        f"[{res.model_name} | {res.split}] "
        f"acc={res.accuracy:.4f}  prec={res.precision:.4f}  rec={res.recall:.4f}  f1={res.f1:.4f}  "
        f"train_time={res.train_time_sec:.3f}s"
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(res.conf_mat)
    print("-" * 70)


def results_to_table(rows: List[EvalResult]):
    if pd is None:
        # minimal fallback
        return None

    data = []
    for r in rows:
        data.append(
            {
                "model": r.model_name,
                "split": r.split,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "train_time_sec": r.train_time_sec,
                "tn": int(r.conf_mat[0, 0]),
                "fp": int(r.conf_mat[0, 1]),
                "fn": int(r.conf_mat[1, 0]),
                "tp": int(r.conf_mat[1, 1]),
            }
        )
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Person 3 — TF-IDF Classical ML Model Comparison")
    parser.add_argument("--tfidf_pkl", default="outputs/tfidf_features.pkl", help="Path to tfidf_features.pkl")
    parser.add_argument("--clean_csv", default="outputs/clean_reviews.csv", help="Path to clean_reviews.csv (label fallback)")
    parser.add_argument("--outputs_dir", default="outputs", help="Where to write result CSV")
    args = parser.parse_args()

    _ensure_outputs_dir(args.outputs_dir)

    X_train, X_val, X_test, y_train, y_val, y_test = load_tfidf_bundle(
        args.tfidf_pkl, clean_csv_path=args.clean_csv
    )

    all_results: List[EvalResult] = []
    best_models: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    # ---------------------------
    # 1) Logistic Regression (tuned)
    # ---------------------------
    print("\n=== Logistic Regression (TF-IDF) ===")
    def lr_ctor(**p):
        return LogisticRegression(
            max_iter=2000,
            solver="liblinear",  # stable for sparse + L1/L2
            random_state=RANDOM_STATE,
            **p,
        )

    lr_grid = {
        "C": [0.25, 1.0, 4.0],
        "penalty": ["l2"],
    }
    lr_model, lr_params, lr_time, lr_val = fit_with_val_tuning(
        lr_ctor, lr_grid, X_train, y_train, X_val, y_val, "LogReg"
    )
    best_models["LogReg"] = (lr_model, lr_params)
    print(f"Best params: {lr_params}")
    # Evaluate train/val/test for overfitting analysis
    lr_train_res = evaluate(lr_model, X_train, y_train, "train", lr_time, "LogReg")
    lr_val_res = lr_val
    lr_test_res = evaluate(lr_model, X_test, y_test, "test", lr_time, "LogReg")
    for r in [lr_train_res, lr_val_res, lr_test_res]:
        print_result(r)
        all_results.append(r)

    # ---------------------------
    # 2) Linear SVM (tuned)
    # ---------------------------
    print("\n=== Linear SVM (TF-IDF) ===")
    def svm_ctor(**p):
        return LinearSVC(random_state=RANDOM_STATE, **p)

    svm_grid = {
        "C": [0.25, 1.0, 4.0],
    }
    svm_model, svm_params, svm_time, svm_val = fit_with_val_tuning(
        svm_ctor, svm_grid, X_train, y_train, X_val, y_val, "LinearSVM"
    )
    best_models["LinearSVM"] = (svm_model, svm_params)
    print(f"Best params: {svm_params}")
    svm_train_res = evaluate(svm_model, X_train, y_train, "train", svm_time, "LinearSVM")
    svm_val_res = svm_val
    svm_test_res = evaluate(svm_model, X_test, y_test, "test", svm_time, "LinearSVM")
    for r in [svm_train_res, svm_val_res, svm_test_res]:
        print_result(r)
        all_results.append(r)

    # ---------------------------
    # 3) Random Forest (TF-IDF via SVD to avoid dense blowup)
    # ---------------------------
    print("\n=== Random Forest (TF-IDF + TruncatedSVD) ===")
    # Trees generally dislike ultra-high-dimensional sparse TF-IDF, so we reduce dimension first.
    rf_pipeline = lambda n_components, n_estimators, max_depth: Pipeline(
        steps=[
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    rf_grid = {
        "n_components": [200, 400],
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
    }

    # custom tuning loop for pipeline
    best_rf = None
    best_rf_params = None
    best_rf_val = None
    best_rf_time = None

    for n_components in rf_grid["n_components"]:
        for n_estimators in rf_grid["n_estimators"]:
            for max_depth in rf_grid["max_depth"]:
                model = rf_pipeline(n_components, n_estimators, max_depth)
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                t1 = time.perf_counter()
                tr_time = t1 - t0
                val_res = evaluate(model, X_val, y_val, "val", tr_time, "RandomForest")
                if (best_rf_val is None) or (val_res.f1 > best_rf_val.f1) or (
                    np.isclose(val_res.f1, best_rf_val.f1) and val_res.accuracy > best_rf_val.accuracy
                ):
                    best_rf = model
                    best_rf_params = {
                        "n_components": n_components,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                    }
                    best_rf_val = val_res
                    best_rf_time = tr_time

    assert best_rf is not None and best_rf_params is not None and best_rf_val is not None and best_rf_time is not None
    best_models["RandomForest"] = (best_rf, best_rf_params)
    print(f"Best params: {best_rf_params}")
    rf_train_res = evaluate(best_rf, X_train, y_train, "train", best_rf_time, "RandomForest")
    rf_val_res = best_rf_val
    rf_test_res = evaluate(best_rf, X_test, y_test, "test", best_rf_time, "RandomForest")
    for r in [rf_train_res, rf_val_res, rf_test_res]:
        print_result(r)
        all_results.append(r)

    # ---------------------------
    # 4) XGBoost (tuned; handles sparse TF-IDF)
    # ---------------------------
    print("\n=== XGBoost (TF-IDF) ===")
    if not _HAS_XGB:
        print("XGBoost is not installed. Skipping XGBoost model.\n"
              "To enable: pip install xgboost")
    else:
        def xgb_ctor(**p):
            # Use conservative params; fast + stable baseline
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",   # good default (works CPU). If GPU available you can change to 'gpu_hist'.
                **p,
            )

        xgb_grid = {
            "n_estimators": [300, 600],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }

        # manual grid to keep tuning explainable
        best_xgb = None
        best_xgb_params = None
        best_xgb_val = None
        best_xgb_time = None

        for n_estimators in xgb_grid["n_estimators"]:
            for lr in xgb_grid["learning_rate"]:
                for md in xgb_grid["max_depth"]:
                    for subs in xgb_grid["subsample"]:
                        for cols in xgb_grid["colsample_bytree"]:
                            model = xgb_ctor(
                                n_estimators=n_estimators,
                                learning_rate=lr,
                                max_depth=md,
                                subsample=subs,
                                colsample_bytree=cols,
                                reg_lambda=1.0,
                            )
                            t0 = time.perf_counter()
                            model.fit(X_train, y_train)
                            t1 = time.perf_counter()
                            tr_time = t1 - t0

                            val_res = evaluate(model, X_val, y_val, "val", tr_time, "XGBoost")
                            if (best_xgb_val is None) or (val_res.f1 > best_xgb_val.f1) or (
                                np.isclose(val_res.f1, best_xgb_val.f1) and val_res.accuracy > best_xgb_val.accuracy
                            ):
                                best_xgb = model
                                best_xgb_params = {
                                    "n_estimators": n_estimators,
                                    "learning_rate": lr,
                                    "max_depth": md,
                                    "subsample": subs,
                                    "colsample_bytree": cols,
                                }
                                best_xgb_val = val_res
                                best_xgb_time = tr_time

        assert best_xgb is not None and best_xgb_params is not None and best_xgb_val is not None and best_xgb_time is not None
        best_models["XGBoost"] = (best_xgb, best_xgb_params)
        print(f"Best params: {best_xgb_params}")
        xgb_train_res = evaluate(best_xgb, X_train, y_train, "train", best_xgb_time, "XGBoost")
        xgb_val_res = best_xgb_val
        xgb_test_res = evaluate(best_xgb, X_test, y_test, "test", best_xgb_time, "XGBoost")
        for r in [xgb_train_res, xgb_val_res, xgb_test_res]:
            print_result(r)
            all_results.append(r)

    # ---------------------------
    # Create comparison table on TEST (as final ranking)
    # ---------------------------
    print("\n=== Model Comparison (Test Split) ===")
    test_rows = [r for r in all_results if r.split == "test"]

    # Choose best by test F1; tie-breaker by accuracy, then by training time
    def key_fn(r: EvalResult):
        return (r.f1, r.accuracy, -r.train_time_sec)

    best_test = sorted(test_rows, key=key_fn, reverse=True)[0]
    print(f"Best TF-IDF model (by test F1): {best_test.model_name} | "
          f"F1={best_test.f1:.4f}, Acc={best_test.accuracy:.4f}")

    if pd is not None:
        df = results_to_table(all_results)
        assert df is not None
        out_csv = os.path.join(args.outputs_dir, "tfidf_model_comparison.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved full comparison table to: {out_csv}")

        # Print a compact view for the report (only TEST rows)
        compact = df[df["split"] == "test"][["model", "accuracy", "precision", "recall", "f1", "train_time_sec"]]
        compact = compact.sort_values(by=["f1", "accuracy"], ascending=False)
        print("\nTest split summary:")
        print(compact.to_string(index=False))
    else:
        print("pandas not installed; skipping CSV table output.")

    # ---------------------------
    # Overfitting analysis (train vs val vs test gaps)
    # ---------------------------
    print("\n=== Overfitting Analysis (Train vs Val vs Test F1) ===")
    by_model = {}
    for r in all_results:
        by_model.setdefault(r.model_name, {})[r.split] = r

    for model_name, splits in by_model.items():
        if not all(s in splits for s in ["train", "val", "test"]):
            continue
        tr = splits["train"].f1
        va = splits["val"].f1
        te = splits["test"].f1
        gap_tv = tr - va
        gap_tt = tr - te
        print(
            f"{model_name:12s}  "
            f"F1(train/val/test) = {tr:.4f}/{va:.4f}/{te:.4f}  "
            f"gap(train-val)={gap_tv:+.4f}  gap(train-test)={gap_tt:+.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()