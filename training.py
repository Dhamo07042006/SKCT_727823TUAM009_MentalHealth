import argparse
import json
import os
import tempfile
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def configure_mlflow_tracking() -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    local_mlruns = os.path.abspath("mlruns")
    os.makedirs(local_mlruns, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{local_mlruns}")


def find_dataset_path(user_path: str | None) -> str:
    candidates: list[str] = []
    if user_path:
        candidates.append(user_path)

    env_path = os.environ.get("DATA_PATH")
    if env_path:
        candidates.append(env_path)

    candidates.extend(
        [
            os.path.join("data", "raw", "student_mental_health.csv"),
            os.path.join("data", "raw", "Student Mental health.csv"),
            os.path.join("data", "raw", "Student Mental health.csv "),
            os.path.join("data", "raw", "Student_Mental_health.csv"),
            os.path.join("data", "raw", "Student Mental Health.csv"),
            os.path.join("data", "raw", "Student Mental Health Dataset.csv"),
            os.path.join("data", "raw", "Student Mental Health.csv"),
            os.path.join("student_mental_health.csv"),
            os.path.join("Student Mental health.csv"),
        ]
    )

    for p in candidates:
        if p and os.path.exists(p) and os.path.isfile(p):
            return p

    raw_dir = os.path.join("data", "raw")
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.lower().endswith(".csv"):
                return os.path.join(raw_dir, f)

    raise FileNotFoundError(
        "Dataset CSV not found. Provide --data_path or set DATA_PATH, "
        "or place the Kaggle CSV under data/raw/."
    )


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


def choose_target_column(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target and user_target in df.columns:
        return user_target

    preferred = [
        "Depression",
        "Anxiety",
        "Panic attack",
        "Treatment",
        "mental_health_condition",
        "condition",
        "target",
        "label",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    return df.columns[-1]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def plot_eda(df: pd.DataFrame, target_col: str, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    hist_col = "Age" if "Age" in df.columns else (num_cols[0] if num_cols else None)
    if hist_col is not None:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[hist_col].dropna(), kde=True)
        plt.title(f"Histogram: {hist_col}")
        p = os.path.join(out_dir, "histogram.png")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths["histogram"] = p

    if len(num_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        p = os.path.join(out_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths["correlation_heatmap"] = p

    if target_col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[target_col].astype(str))
        plt.title(f"Countplot: {target_col}")
        plt.xticks(rotation=30, ha="right")
        p = os.path.join(out_dir, "countplot.png")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths["countplot"] = p

    return paths


def safe_roc_auc(y_true, y_score, n_classes: int) -> float | None:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score, multi_class="ovo", average="weighted"))
    except Exception:
        return None


def get_pos_label(y: pd.Series) -> str:
    labels = sorted(pd.Series(y).dropna().astype(str).unique().tolist())
    if "Yes" in labels:
        return "Yes"
    if len(labels) >= 2:
        return labels[1]
    return "1"


def model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(model, tmp_path)
        size_bytes = os.path.getsize(tmp_path)
        return float(size_bytes / (1024 * 1024))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_n_features(preprocessor: ColumnTransformer) -> int:
    try:
        feature_names = preprocessor.get_feature_names_out()
        return int(len(feature_names))
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--roll_no", type=str, default="123")
    parser.add_argument("--experiment_name", type=str, default="SKCT_727823TUAM009_MentalHealth")
    args = parser.parse_args()

    configure_mlflow_tracking()
    mlflow.set_experiment(args.experiment_name)

    data_path = find_dataset_path(args.data_path)
    df = load_dataset(data_path)

    target_col = choose_target_column(df, args.target)

    df = df.dropna(subset=[target_col])
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    eda_dir = os.path.join("artifacts", "eda")
    eda_paths = plot_eda(df, target_col=target_col, out_dir=eda_dir)

    experiment_configs: list[dict] = [
        {"model": "logreg", "params": {"C": 1.0, "max_iter": 1000}, "seed": 7},
        {"model": "logreg", "params": {"C": 0.3, "max_iter": 2000}, "seed": 7},
        {"model": "logreg", "params": {"C": 3.0, "max_iter": 2000}, "seed": 21},
        {"model": "rf", "params": {"n_estimators": 200, "max_depth": None}, "seed": 7},
        {"model": "rf", "params": {"n_estimators": 400, "max_depth": 10}, "seed": 7},
        {"model": "rf", "params": {"n_estimators": 600, "max_depth": 20}, "seed": 21},
        {"model": "svm", "params": {"C": 1.0, "kernel": "rbf", "probability": True}, "seed": 7},
        {"model": "svm", "params": {"C": 0.5, "kernel": "linear", "probability": True}, "seed": 7},
        {"model": "svm", "params": {"C": 2.0, "kernel": "rbf", "gamma": "scale", "probability": True}, "seed": 21},
        {"model": "knn", "params": {"n_neighbors": 5, "weights": "uniform"}, "seed": 7},
        {"model": "knn", "params": {"n_neighbors": 11, "weights": "distance"}, "seed": 7},
        {"model": "dt", "params": {"max_depth": None, "min_samples_split": 2}, "seed": 7},
        {"model": "dt", "params": {"max_depth": 8, "min_samples_split": 10}, "seed": 21},
        {"model": "dt", "params": {"max_depth": 4, "min_samples_split": 20}, "seed": 42},
    ]

    best = {
        "f1": -1.0,
        "run_id": None,
        "model": None,
        "metrics": None,
        "params": None,
    }

    os.makedirs("models", exist_ok=True)

    for i, cfg in enumerate(experiment_configs, start=1):
        seed = int(cfg["seed"])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=seed,
            stratify=y if y.nunique() > 1 else None,
        )

        preprocessor = make_preprocessor(X_train)

        model_name = cfg["model"]
        model_params = cfg["params"].copy()

        if model_name == "logreg":
            clf = LogisticRegression(
                random_state=seed,
                solver="lbfgs",
                **model_params,
            )
        elif model_name == "rf":
            clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **model_params)
        elif model_name == "svm":
            clf = SVC(random_state=seed, **model_params)
        elif model_name == "knn":
            clf = KNeighborsClassifier(**model_params)
        elif model_name == "dt":
            clf = DecisionTreeClassifier(random_state=seed, **model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

        run_name = f"exp_{i:02d}_{model_name}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("run_index", i)
            mlflow.log_param("model", model_name)
            mlflow.log_param("random_seed", seed)
            mlflow.log_param("target_col", target_col)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("test_size", args.test_size)
            for k, v in model_params.items():
                mlflow.log_param(k, v)

            start = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            y_pred = pipe.predict(X_test)

            is_binary = y.nunique() == 2
            avg = "binary" if is_binary else "weighted"
            pos_label = get_pos_label(y) if is_binary else None

            if is_binary:
                f1 = float(
                    f1_score(
                        y_test,
                        y_pred,
                        average=avg,
                        pos_label=pos_label,
                        zero_division=0,
                    )
                )
                precision = float(
                    precision_score(
                        y_test,
                        y_pred,
                        average=avg,
                        pos_label=pos_label,
                        zero_division=0,
                    )
                )
                recall = float(
                    recall_score(
                        y_test,
                        y_pred,
                        average=avg,
                        pos_label=pos_label,
                        zero_division=0,
                    )
                )
            else:
                f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
                precision = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
                recall = float(recall_score(y_test, y_pred, average=avg, zero_division=0))

            n_classes = int(y.nunique())
            y_score = None
            if hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(X_test)
                    if n_classes == 2:
                        y_score = proba[:, 1]
                    else:
                        y_score = proba
                except Exception:
                    y_score = None
            if y_score is None and hasattr(pipe, "decision_function"):
                try:
                    df_score = pipe.decision_function(X_test)
                    y_score = df_score
                except Exception:
                    y_score = None

            if y_score is not None:
                if n_classes == 2 and pos_label is not None:
                    y_bin = (pd.Series(y_test).astype(str) == str(pos_label)).astype(int)
                    roc_auc = safe_roc_auc(y_bin, y_score, n_classes)
                else:
                    roc_auc = safe_roc_auc(y_test, y_score, n_classes)
            else:
                roc_auc = None

            fitted_preprocessor = pipe.named_steps["preprocess"]
            n_feats = get_n_features(fitted_preprocessor)

            size_mb = model_size_mb(pipe)

            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", float(roc_auc))
            mlflow.log_metric("training_time_seconds", float(train_time))
            mlflow.log_metric("model_size_mb", float(size_mb))
            mlflow.log_metric("random_seed", float(seed))
            mlflow.log_metric("n_features", float(n_feats))

            for _, p in eda_paths.items():
                if os.path.exists(p):
                    mlflow.log_artifact(p, artifact_path="eda")

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            if f1 > best["f1"]:
                best["f1"] = f1
                best["run_id"] = mlflow.active_run().info.run_id
                best["model"] = pipe
                best["metrics"] = {
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    "training_time_seconds": float(train_time),
                    "model_size_mb": float(size_mb),
                    "random_seed": seed,
                    "n_features": n_feats,
                }
                best["params"] = {"model": model_name, **model_params}

    if best["model"] is None:
        raise RuntimeError("No model trained.")

    best_model_path = os.path.join("models", "best_model.joblib")
    joblib.dump(best["model"], best_model_path)

    best_info_path = os.path.join("models", "best_model_info.json")
    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": _now_iso(),
                "experiment": args.experiment_name,
                "best_run_id": best["run_id"],
                "best_params": best["params"],
                "best_metrics": best["metrics"],
                "data_path": data_path,
                "target_col": target_col,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
