"""Log training runs to the SageMaker managed MLflow tracking server.

The sagemaker-mlflow package registers itself with MLflow, so an ARN works as a
tracking URI and every request carries a SigV4 signature. There is no MLflow
user and no password.

Tracking is optional. If the server is stopped or absent, training still runs
and prints the reason. A demo must not stop because a tracking server is down.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


class Tracker:
    """A thin wrapper over the MLflow fluent API.

    Use `Tracker.connect()`, which returns None when tracking is not available.
    """

    def __init__(self, mlflow_module, uri: str) -> None:
        self.mlflow = mlflow_module
        self.uri = uri
        self._run_ids: list[str] = []

    @classmethod
    def connect(cls, quiet: bool = False) -> "Tracker | None":
        """Return a Tracker, or None when tracking is not available.

        The status check uses boto3, and it answers in well under a second. The
        MLflow client itself retries with a backoff, and it needs more than four
        minutes to report that a server is absent. A demo cannot wait that long,
        so the fast check comes first.
        """
        try:
            import boto3
            import botocore
        except ImportError:  # pragma: no cover
            return None

        sm = boto3.client("sagemaker", region_name=config.REGION)
        try:
            info = sm.describe_mlflow_tracking_server(
                TrackingServerName=config.MLFLOW_SERVER_NAME
            )
            status = info["TrackingServerStatus"]
        except botocore.exceptions.ClientError:
            if not quiet:
                print(f"  no MLflow server named {config.MLFLOW_SERVER_NAME}. "
                      "Training continues without tracking.")
                print("  Create it:  uv run python ml/mlflow_server.py create")
            return None

        if status != "Created":
            if not quiet:
                print(f"  the MLflow server is {status}, not running. "
                      "Training continues without tracking.")
                print("  Start it:  uv run python ml/mlflow_server.py start")
            return None

        try:
            import mlflow
        except ImportError:
            if not quiet:
                print("  mlflow is not installed. Skipping experiment tracking.")
            return None

        # Bound the retries anyway, so a server that stops mid-run fails fast.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

        uri = config.mlflow_tracking_uri()
        try:
            mlflow.set_tracking_uri(uri)
            # search_experiments is the cheapest call that proves both the
            # network path and the SigV4 signature work.
            mlflow.search_experiments(max_results=1)
        except Exception as err:  # noqa: BLE001
            if not quiet:
                print(f"  MLflow tracking is not available: {type(err).__name__}")
                print("  Training continues without tracking.")
            return None

        print(f"  tracking to {uri}")
        return cls(mlflow, uri)

    def _use_experiment(self, name: str) -> None:
        """Select the experiment, and restore it first if somebody removed it.

        MLflow removes an experiment softly. The name stays reserved, and
        set_experiment then fails with a message about a deleted experiment.
        Restoring it is the behavior a demo wants.
        """
        client = self.mlflow.tracking.MlflowClient()
        existing = client.get_experiment_by_name(name)
        if existing is not None and existing.lifecycle_stage == "deleted":
            print(f"  restoring the removed experiment {name}")
            client.restore_experiment(existing.experiment_id)
        self.mlflow.set_experiment(name)

    def log_run(self, domain, feature_set: str, metrics: dict,
                scorecard: dict, features: list[str]) -> None:
        """One MLflow run for one feature set."""
        self._use_experiment(domain.experiment)
        with self.mlflow.start_run(run_name=feature_set) as run:
            self.mlflow.log_params({
                "model": "logistic_regression",
                "feature_set": feature_set,
                "regularization_c": metrics["regularization_c"]
                if "regularization_c" in metrics else 1.0,
                "max_iter": 2000,
                "test_size": 0.25,
                "n_features": len(features),
                "target": domain.target,
                "database": domain.database,
            })
            self.mlflow.log_metrics({
                "test_auc": metrics["test_auc"],
                "train_auc": metrics["train_auc"],
                "base_rate": metrics["base_rate"],
                "n_train": metrics["n_train"],
                "n_features": metrics["n_features"],
            })
            self.mlflow.set_tags({
                "company": domain.company,
                "demo_domain": domain.key,
                "endpoint": domain.endpoint,
            })
            # The model is a JSON scorecard, not an MLflow model flavor, so it
            # is logged as an artifact.
            self.mlflow.log_dict(scorecard, "scorecard.json")
            self._run_ids.append(run.info.run_id)

    def log_best(self, domain, scorecard: dict) -> None:
        """Tag the winning run, so the deployed model is easy to find."""
        if not self._run_ids:
            return
        best_set = scorecard["metrics"]["feature_set"]
        client = self.mlflow.tracking.MlflowClient()
        for run_id in self._run_ids:
            run = client.get_run(run_id)
            is_best = run.data.params.get("feature_set") == best_set
            client.set_tag(run_id, "deployed", str(is_best).lower())
        self._run_ids = []

    def runs(self, domain) -> "object":
        """Every run in this demo's experiment, newest first.

        Returns a pandas DataFrame with one row for each run.
        """
        experiment = self.mlflow.get_experiment_by_name(domain.experiment)
        if experiment is None:
            return None
        frame = self.mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_auc DESC"],
        )
        if frame.empty:
            return frame
        keep = [c for c in [
            "tags.mlflow.runName", "params.feature_set", "params.n_features",
            "metrics.test_auc", "metrics.train_auc", "tags.deployed",
            "start_time",
        ] if c in frame.columns]
        tidy = frame[keep].rename(columns=lambda c: c.split(".")[-1])
        return tidy

    def ui_url(self) -> str | None:
        """A signed link that opens the MLflow UI in a browser."""
        import boto3

        sm = boto3.client("sagemaker", region_name=config.REGION)
        try:
            return sm.create_presigned_mlflow_tracking_server_url(
                TrackingServerName=config.MLFLOW_SERVER_NAME
            )["AuthorizedUrl"]
        except Exception:  # noqa: BLE001
            return None
