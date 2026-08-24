"""AWS resource names for the demo, in one place.

This repository holds two demos. One is for financial services, Aurora Lending
Group. The other is for life sciences, Helix Therapeutics. They are
alternatives for different sessions, so select one. Every script takes an
explicit ``--domain finance|lifesci``, and nothing runs both by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- AWS -------------------------------------------------------------------
REGION = os.environ.get("POSIT_DEMO_REGION", "us-east-2")


def account_id() -> str:
    """The AWS account that the demo runs in.

    The account comes from AWS STS. No account number is therefore written into
    this public repository, and a fork works with no edit. To avoid the STS
    call, set POSIT_DEMO_ACCOUNT_ID.
    """
    from_env = os.environ.get("POSIT_DEMO_ACCOUNT_ID")
    if from_env:
        return from_env

    global _ACCOUNT_ID_CACHE
    if _ACCOUNT_ID_CACHE is None:
        import boto3

        _ACCOUNT_ID_CACHE = boto3.client(
            "sts", region_name=REGION
        ).get_caller_identity()["Account"]
    return _ACCOUNT_ID_CACHE


_ACCOUNT_ID_CACHE: str | None = None


BUCKET_PREFIX = "sagemaker-posit-conf-2026-demo"


def bucket() -> str:
    """The S3 bucket that holds the demo data.

    The name starts with "sagemaker" for a reason. AmazonSageMakerFullAccess
    grants S3 object access on arn:aws:s3:::*sagemaker*. The Studio execution
    role therefore needs no S3 policy for this bucket.
    """
    override = os.environ.get("POSIT_DEMO_BUCKET")
    if override:
        return override
    return f"{BUCKET_PREFIX}-{account_id()}-{REGION}"


# Workgroup "primary" has EnforceWorkGroupConfiguration=false and no
# OutputLocation, so every client must pass staging explicitly.
ATHENA_WORKGROUP = os.environ.get("POSIT_DEMO_WORKGROUP", "primary")


def athena_staging() -> str:
    return f"s3://{bucket()}/athena-query-results/"


# The endpoints run on the AWS scikit-learn container. AWS publishes the
# registry account for this image, and it is different in each region.
SKLEARN_REGISTRY_ACCOUNT = {
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-2": "246618743249",
    "eu-west-1": "141502667606",
    "eu-central-1": "492215442770",
}


def serving_image() -> str:
    registry = SKLEARN_REGISTRY_ACCOUNT.get(REGION)
    if registry is None:
        raise SystemExit(
            f"No scikit-learn container registry recorded for {REGION}. "
            "Add it to SKLEARN_REGISTRY_ACCOUNT in config.py."
        )
    return (
        f"{registry}.dkr.ecr.{REGION}.amazonaws.com"
        "/sagemaker-scikit-learn:1.2-1-cpu-py3"
    )


ENDPOINT_INSTANCE_TYPE = os.environ.get("POSIT_DEMO_INSTANCE_TYPE", "ml.m5.large")


@dataclass(frozen=True)
class Domain:
    """The values that are different between the two demos."""

    key: str  # CLI value: finance | lifesci
    company: str  # fictional company name
    industry: str  # for the README disclaimer
    database: str  # Glue database
    fact_table: str  # the table the walkthrough queries
    target: str  # binary column the model predicts
    endpoint: str  # SageMaker endpoint name
    report: str  # path to the Quarto artifact


FINANCE = Domain(
    key="finance",
    company="Aurora Lending Group",
    industry="financial services",
    database="aurora_lending",
    fact_table="fct_loan_performance",
    target="charged_off",
    endpoint="aurora-lending-risk",
    report="reports/aurora_lending/portfolio_risk_review.qmd",
)

LIFESCI = Domain(
    key="lifesci",
    company="Helix Therapeutics",
    industry="life sciences",
    database="helix_trials",
    fact_table="fct_visit_observations",
    target="discontinued",
    endpoint="helix-dropout-risk",
    report="reports/helix_trials/enrollment_safety_review.qmd",
)

DOMAINS = {d.key: d for d in (FINANCE, LIFESCI)}


def get_domain(key: str) -> Domain:
    try:
        return DOMAINS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown domain {key!r}. Choose one of: {', '.join(DOMAINS)}"
        ) from None


def s3_prefix(domain: Domain, *parts: str) -> str:
    """s3:// URI under this domain's namespace."""
    return "/".join([f"s3://{bucket()}", domain.database, *parts])
