# Name RollNo
import argparse
import os
import time
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def configure_mlflow_tracking() -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    local_mlruns = os.path.abspath("mlruns")
    os.makedirs(local_mlruns, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{local_mlruns}")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )


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
    parser.add_argument("--input", type=str, default=os.path.join("data", "processed", "processed.csv"))
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--model_out", type=str, default=os.path.join("models", "pipeline_model.joblib"))
    parser.add_argument("--experiment_name", type=str, default="SKCT_727823TUAM009_MentalHealth")
    args = parser.parse_args()

    print(args.roll_no)
    print(_ts())

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Processed data not found: {args.input}. Run data_prep.py first.")

    configure_mlflow_tracking()
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.input)
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

    pre = make_preprocessor(X_train)
    clf = RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
    pipe = Pipeline([("preprocess", pre), ("model", clf)])

    with mlflow.start_run(run_name="pipeline_train"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("max_depth", 20)
        mlflow.log_param("random_seed", 42)
        mlflow.log_param("target_col", target_col)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        mlflow.log_metric("training_time_seconds", float(train_time))

        mlflow.sklearn.log_model(pipe, artifact_path="model")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)


if __name__ == "__main__":
    main()
