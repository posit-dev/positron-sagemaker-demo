"""Manage the SageMaker managed MLflow tracking server.

    uv run python ml/mlflow_server.py status
    uv run python ml/mlflow_server.py create
    uv run python ml/mlflow_server.py start
    uv run python ml/mlflow_server.py stop
    uv run python ml/mlflow_server.py url
    uv run python ml/mlflow_server.py delete

Cost. A Small tracking server bills $0.60 each hour that it runs, in
us-east-2. Small is the smallest size that AWS offers.

Stop the server after a session. A stopped server does not bill for compute,
and it keeps every run. Storage costs $0.10 for each GB in a month, and the
runs in this demo are a few kilobytes. Use `delete` only to remove the server
and its history for good.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import boto3
import botocore

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

# The tracking server writes artifacts to S3 under this role. The demo reuses
# the Studio execution role: sagemaker.amazonaws.com can already assume it, and
# AmazonSageMakerFullAccess grants S3 access to a bucket whose name holds
# "sagemaker". No new role is necessary.
DEFAULT_ROLE = "sagemaker-demo-execution-role"

READY = "Created"
STOPPED = "Stopped"
IN_PROGRESS = ("Creating", "Starting", "Stopping", "Updating", "Deleting", "Maintenance")


def describe(sm, name: str) -> dict | None:
    try:
        return sm.describe_mlflow_tracking_server(TrackingServerName=name)
    except botocore.exceptions.ClientError as err:
        if err.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return None
        raise


def wait_until(sm, name: str, targets: tuple[str, ...], timeout_s: int = 3_600) -> str:
    """Report the status as it changes, until it reaches one of the targets."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        info = describe(sm, name)
        if info is None:
            print(f"    {time.strftime('%H:%M:%S')}  gone")
            return "Absent"
        status = info["TrackingServerStatus"]
        if status != last:
            print(f"    {time.strftime('%H:%M:%S')}  {status}")
            last = status
        if status in targets:
            return status
        if status not in IN_PROGRESS:
            raise SystemExit(f"unexpected status {status}. Read the SageMaker console.")
        time.sleep(20)
    raise SystemExit(
        f"{name} did not reach {targets} within {timeout_s // 60} minutes. "
        f"The last status was {last}."
    )


def cmd_status(sm, name: str) -> int:
    info = describe(sm, name)
    if info is None:
        print(f"{name}: absent")
        print("Create it:  uv run python ml/mlflow_server.py create")
        return 1
    print(f"{name}")
    print(f"  status    {info['TrackingServerStatus']}")
    print(f"  size      {info.get('TrackingServerSize')}")
    print(f"  mlflow    {info.get('MlflowVersion')}")
    print(f"  artifacts {info.get('ArtifactStoreUri')}")
    print(f"  uri       {info['TrackingServerArn']}")
    if info["TrackingServerStatus"] == READY:
        print("\n  This server is running and billing. Stop it after the session:")
        print("    uv run python ml/mlflow_server.py stop")
    return 0


def cmd_create(sm, name: str, role_arn: str, wait: bool) -> int:
    if describe(sm, name) is not None:
        print(f"{name} already exists")
        return cmd_status(sm, name)
    print(f"creating {name} ({config.MLFLOW_SERVER_SIZE}) in {config.REGION}")
    print(f"  artifacts {config.mlflow_artifact_store()}")
    print(f"  role      {role_arn}")
    sm.create_mlflow_tracking_server(
        TrackingServerName=name,
        ArtifactStoreUri=config.mlflow_artifact_store(),
        TrackingServerSize=config.MLFLOW_SERVER_SIZE,
        RoleArn=role_arn,
        # No automatic model registration. This demo logs a JSON scorecard, not
        # an MLflow model flavor, so the option would do nothing.
        AutomaticModelRegistration=False,
    )
    print("  creation takes about 22 minutes")
    if wait:
        wait_until(sm, name, (READY,))
        return cmd_status(sm, name)
    return 0


def cmd_start(sm, name: str, wait: bool) -> int:
    info = describe(sm, name)
    if info is None:
        raise SystemExit(f"{name} does not exist. Run: ml/mlflow_server.py create")
    if info["TrackingServerStatus"] == READY:
        print(f"{name} is already running")
        return 0
    print(f"starting {name}")
    sm.start_mlflow_tracking_server(TrackingServerName=name)
    if wait:
        wait_until(sm, name, (READY,))
    return 0


def cmd_stop(sm, name: str, wait: bool) -> int:
    info = describe(sm, name)
    if info is None:
        print(f"{name} does not exist. Nothing to stop.")
        return 0
    if info["TrackingServerStatus"] == STOPPED:
        print(f"{name} is already stopped")
        return 0
    print(f"stopping {name}. The run history stays.")
    sm.stop_mlflow_tracking_server(TrackingServerName=name)
    if wait:
        wait_until(sm, name, (STOPPED,))
    return 0


def cmd_delete(sm, name: str) -> int:
    if describe(sm, name) is None:
        print(f"{name} does not exist")
        return 0
    print(f"deleting {name}. This removes every run.")
    sm.delete_mlflow_tracking_server(TrackingServerName=name)
    return 0


def cmd_url(sm, name: str) -> int:
    info = describe(sm, name)
    if info is None or info["TrackingServerStatus"] != READY:
        raise SystemExit(
            f"{name} is not running. Start it: ml/mlflow_server.py start"
        )
    presigned = sm.create_presigned_mlflow_tracking_server_url(TrackingServerName=name)
    print(presigned["AuthorizedUrl"])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["status", "create", "start", "stop", "delete", "url"]
    )
    parser.add_argument("--name", default=config.MLFLOW_SERVER_NAME)
    parser.add_argument("--role-arn", default=None, help="role for the tracking server")
    parser.add_argument(
        "--no-wait", action="store_true", help="return before the status settles"
    )
    args = parser.parse_args()

    session = boto3.Session(region_name=config.REGION)
    sm = session.client("sagemaker")
    wait = not args.no_wait
    role = args.role_arn or f"arn:aws:iam::{config.account_id()}:role/{DEFAULT_ROLE}"

    actions = {
        "status": lambda: cmd_status(sm, args.name),
        "create": lambda: cmd_create(sm, args.name, role, wait),
        "start": lambda: cmd_start(sm, args.name, wait),
        "stop": lambda: cmd_stop(sm, args.name, wait),
        "delete": lambda: cmd_delete(sm, args.name),
        "url": lambda: cmd_url(sm, args.name),
    }
    raise SystemExit(actions[args.action]())


if __name__ == "__main__":
    main()
