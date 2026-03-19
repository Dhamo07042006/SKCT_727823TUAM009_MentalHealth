# Name RollNo
import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_pos_label(y: pd.Series) -> str:
    labels = sorted(pd.Series(y).dropna().astype(str).unique().tolist())
    if "Yes" in labels:
        return "Yes"
    if len(labels) >= 2:
        return labels[1]
    return "1"


def safe_roc_auc(y_true, y_score, n_classes: int, pos_label: str | None):
    try:
        if n_classes == 2:
            if pos_label is None:
                return float(roc_auc_score(y_true, y_score))
            y_bin = (pd.Series(y_true).astype(str) == str(pos_label)).astype(int)
            return float(roc_auc_score(y_bin, y_score))
        return float(roc_auc_score(y_true, y_score, multi_class="ovo", average="weighted"))
    except Exception:
        return None


def choose_target(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target and user_target in df.columns:
        return user_target
    for c in ["Depression", "Anxiety", "Panic attack", "Treatment", "target", "label"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll_no", type=str, default="123")
    parser.add_argument("--data", type=str, default=os.path.join("data", "processed", "processed.csv"))
    parser.add_argument("--model", type=str, default=os.path.join("models", "pipeline_model.joblib"))
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    print(args.roll_no)
    print(_ts())

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Processed data not found: {args.data}. Run data_prep.py first.")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}. Run train_pipeline.py first.")

    df = pd.read_csv(args.data)
    target_col = choose_target(df, args.target)

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = joblib.load(args.model)

    y_pred = model.predict(X_test)
    is_binary = y.nunique() == 2
    avg = "binary" if is_binary else "weighted"
    pos_label = get_pos_label(y) if is_binary else None

    if is_binary:
        f1 = float(f1_score(y_test, y_pred, average=avg, pos_label=pos_label, zero_division=0))
        precision = float(
            precision_score(y_test, y_pred, average=avg, pos_label=pos_label, zero_division=0)
        )
        recall = float(recall_score(y_test, y_pred, average=avg, pos_label=pos_label, zero_division=0))
    else:
        f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
        precision = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
        recall = float(recall_score(y_test, y_pred, average=avg, zero_division=0))

    n_classes = int(y.nunique())
    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                y_score = proba[:, 1]
            else:
                y_score = proba
        except Exception:
            y_score = None

    if y_score is None and hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = None

    roc_auc = safe_roc_auc(y_test, y_score, n_classes, pos_label) if y_score is not None else None

    print({
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    })


if __name__ == "__main__":
    main()
