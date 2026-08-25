"""Stop everything that bills, after a session.

    uv run python ml/teardown.py --domain finance
    uv run python ml/teardown.py --all

A real-time endpoint bills for every hour that it exists, even when nothing
calls it. This script removes the endpoints.

With --all it also stops the MLflow tracking server, which bills $0.60 each
hour that it runs. The server is stopped and not removed, so every run stays.
To remove the server and its history, run:

    uv run python ml/mlflow_server.py delete

You can run this script more than once. If a resource is already absent, the
script reports that and continues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
import botocore

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


def _delete(label: str, fn, **kwargs) -> None:
    try:
        fn(**kwargs)
        print(f"  deleted  {label}")
    except botocore.exceptions.ClientError as err:
        if err.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            print(f"  absent   {label}")
        else:
            raise


def teardown(domain: config.Domain, sm) -> None:
    print(f"{domain.company}: endpoint {domain.endpoint}")

    # Read the endpoint config name before the endpoint is removed.
    config_name = None
    try:
        config_name = sm.describe_endpoint(EndpointName=domain.endpoint)["EndpointConfigName"]
    except botocore.exceptions.ClientError:
        pass

    _delete(f"endpoint {domain.endpoint}", sm.delete_endpoint, EndpointName=domain.endpoint)

    # Each deploy leaves an endpoint config and a model, both named with a
    # timestamp. Remove all of them, so that repeated preparation for a session
    # does not leave unused resources behind.
    configs = sm.list_endpoint_configs(NameContains=domain.endpoint)["EndpointConfigs"]
    for c in configs:
        _delete(
            f"endpoint-config {c['EndpointConfigName']}",
            sm.delete_endpoint_config,
            EndpointConfigName=c["EndpointConfigName"],
        )
    if config_name and config_name not in {c["EndpointConfigName"] for c in configs}:
        _delete(f"endpoint-config {config_name}", sm.delete_endpoint_config,
                EndpointConfigName=config_name)

    for m in sm.list_models(NameContains=domain.endpoint)["Models"]:
        _delete(f"model {m['ModelName']}", sm.delete_model, ModelName=m["ModelName"])


def stop_mlflow(sm) -> None:
    """Stop the tracking server. The runs stay, and the compute stops billing."""
    name = config.MLFLOW_SERVER_NAME
    try:
        status = sm.describe_mlflow_tracking_server(
            TrackingServerName=name)["TrackingServerStatus"]
    except botocore.exceptions.ClientError:
        print(f"MLflow server {name}: absent")
        return

    if status == "Stopped":
        print(f"MLflow server {name}: already stopped")
        return
    if status != "Created":
        print(f"MLflow server {name}: {status}. Leaving it alone.")
        return

    sm.stop_mlflow_tracking_server(TrackingServerName=name)
    print(f"MLflow server {name}: stopping. The run history stays.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", choices=sorted(config.DOMAINS))
    group.add_argument("--all", action="store_true", help="tear down both domains")
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=config.REGION)
    targets = list(config.DOMAINS.values()) if args.all else [config.get_domain(args.domain)]
    for domain in targets:
        teardown(domain, sm)

    if args.all:
        print()
        stop_mlflow(sm)

    remaining = sm.list_endpoints()["Endpoints"]
    print(f"\nendpoints still in {config.REGION}: {[e['EndpointName'] for e in remaining] or 'none'}")


if __name__ == "__main__":
    main()
