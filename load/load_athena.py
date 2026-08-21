"""Publish one domain's synthetic parquet to S3 and register it in Glue.

    uv run python load/load_athena.py --domain finance

Run this from a workstation with write access (PowerUser is enough -- no
iam:PassRole needed). The SageMaker execution role only ever reads.

awswrangler.s3.to_parquet(dataset=True, database=..., table=...) writes the
parquet *and* creates the Glue table in one call, so there is no crawler and no
DDL to maintain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

import config  # noqa: E402
from generate import MONEY_COLUMNS, out_dir_for  # noqa: E402

# Tables are loaded dimensions-first so the catalogue reads sensibly.
TABLE_ORDER = {
    "aurora_lending": ["dim_date", "dim_loan_product", "dim_borrower", "fct_loan_performance"],
    "helix_trials": ["dim_visit", "dim_site", "dim_subject", "fct_visit_observations"],
}


def ensure_bucket(session: boto3.Session) -> None:
    """Create the bucket in the right region, or fail loudly.

    A bucket created without LocationConstraint lands in us-east-1 and cannot
    be moved -- and Athena fails confusingly across regions -- so the region is
    asserted before any data is written.
    """
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=config.bucket())
        print(f"bucket {config.bucket()} exists")
    except s3.exceptions.ClientError as err:
        if err.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        print(f"creating bucket {config.bucket()} in {config.REGION}")
        s3.create_bucket(
            Bucket=config.bucket(),
            CreateBucketConfiguration={"LocationConstraint": config.REGION},
        )

    located = s3.get_bucket_location(Bucket=config.bucket())["LocationConstraint"]
    if located != config.REGION:
        raise SystemExit(
            f"bucket {config.bucket()} is in {located!r}, expected {config.REGION!r}. "
            "Athena cannot query across regions; delete it and re-run."
        )


def ensure_database(database: str) -> None:
    if database not in wr.catalog.databases(limit=1000)["Database"].tolist():
        print(f"creating Glue database {database}")
        wr.catalog.create_database(name=database, exist_ok=True)
    else:
        print(f"Glue database {database} exists")


def athena_dtype_overrides(df: pd.DataFrame) -> dict[str, str]:
    """Pin the columns whose type inference we do not want to trust.

    Money must stay decimal so cents are exact; everything else infers fine.
    """
    return {c: "decimal(12,2)" for c in df.columns if c in MONEY_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(config.DOMAINS))
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    src = out_dir_for(domain, REPO_ROOT / "data")
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Run: uv run python data/generate.py --domain {args.domain}"
        )

    session = boto3.Session(region_name=config.REGION)
    print(f"account {session.client('sts').get_caller_identity()['Account']} / {config.REGION}")
    ensure_bucket(session)
    ensure_database(domain.database)

    for table in TABLE_ORDER[domain.database]:
        parquet = src / table / f"{table}.parquet"
        df = pd.read_parquet(parquet)
        path = config.s3_prefix(domain, table) + "/"
        wr.s3.to_parquet(
            df=df,
            path=path,
            dataset=True,
            mode="overwrite",
            database=domain.database,
            table=table,
            dtype=athena_dtype_overrides(df),
            compression="snappy",
            boto3_session=session,
        )
        print(f"  {table:26} {len(df):>7,} rows -> {path}")

    print(f"\nloaded {domain.company} into Glue database {domain.database}")
    print(f"next: uv run python load/verify_athena.py --domain {args.domain}")


if __name__ == "__main__":
    main()
