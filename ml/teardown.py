"""Delete the hosted endpoints so they stop billing.

    uv run python ml/teardown.py --domain finance
    uv run python ml/teardown.py --all

A real-time endpoint bills for every hour it exists, whether or not anything
invokes it. Run this after the session. Deleting is idempotent -- missing
resources are reported and skipped, not treated as errors.
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
    print(f"{domain.company} -- endpoint {domain.endpoint}")

    # Capture the config/model names before the endpoint disappears.
    config_name = None
    try:
        config_name = sm.describe_endpoint(EndpointName=domain.endpoint)["EndpointConfigName"]
    except botocore.exceptions.ClientError:
        pass

    _delete(f"endpoint {domain.endpoint}", sm.delete_endpoint, EndpointName=domain.endpoint)

    # Every deploy leaves a config + model named after its timestamp; clear all
    # of them so repeated demo prep does not accumulate clutter.
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

    remaining = sm.list_endpoints()["Endpoints"]
    print(f"\nendpoints still in {config.REGION}: {[e['EndpointName'] for e in remaining] or 'none'}")


if __name__ == "__main__":
    main()
