"""
modelling.py - Kriteria 3 (MLflow Project entry point)

Training script yang dijalankan oleh MLflow Project pada CI. Melatih RandomForest
untuk Telco Churn, mencatat metrik + model ke MLflow (local backend store),
lalu mencetak run_id agar artefak/model dapat diproses langkah CI berikutnya
(upload artefak & build Docker image).
"""

import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "telco_churn_preprocessing", "train.csv")
TEST_PATH = os.path.join(os.path.dirname(__file__), "telco_churn_preprocessing", "test.csv")


def main(n_estimators, max_depth):
    # `mlflow run` pre-creates the run inside whichever experiment the CLI
    # picked (--experiment-name/--experiment-id) and injects MLFLOW_RUN_ID.
    # Calling set_experiment() here would point at a different experiment and
    # clash with that run, so only set it when running this script standalone.
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("Telco_Churn_CI")

    train = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]

    # `mlflow run` pre-creates a run and passes its ID via MLFLOW_RUN_ID; call
    # start_run() with no run_name so we resume that run instead of clashing.
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", "ci_rf")
        depth = None if max_depth is not None and max_depth < 0 else max_depth
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=depth, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        print(f"RUN_ID={run_id}")
        # Simpan run_id ke file agar mudah dibaca langkah CI berikutnya.
        with open(os.path.join(os.path.dirname(__file__), "run_id.txt"), "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
