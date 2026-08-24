"""Upload one demo's parquet to S3 and register the tables in AWS Glue.

    uv run python load/load_athena.py --domain finance

Run this from a workstation that has write access. PowerUser is enough, and
iam:PassRole is not necessary. The SageMaker execution role only reads.

One call to awswrangler.s3.to_parquet(dataset=True, database=..., table=...)
writes the parquet and creates the Glue table. There is no crawler, and there
is no DDL to maintain.
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

# Load the dimension tables first, so the catalog reads in a logical order.
TABLE_ORDER = {
    "aurora_lending": ["dim_date", "dim_loan_product", "dim_borrower", "fct_loan_performance"],
    "helix_trials": ["dim_visit", "dim_site", "dim_subject", "fct_visit_observations"],
}


def ensure_bucket(session: boto3.Session) -> None:
    """Create the bucket in the correct region, or stop with an error.

    A bucket made without LocationConstraint goes to us-east-1, and you cannot
    move it. Athena also gives a confusing error across regions. This function
    therefore confirms the region before any data is written.
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
            "Athena cannot query across regions. Remove the bucket and run this again."
        )


def ensure_database(database: str) -> None:
    if database not in wr.catalog.databases(limit=1000)["Database"].tolist():
        print(f"creating Glue database {database}")
        wr.catalog.create_database(name=database, exist_ok=True)
    else:
        print(f"Glue database {database} exists")


def athena_dtype_overrides(df: pd.DataFrame) -> dict[str, str]:
    """Fix the type of the columns where inference is not safe.

    Money must stay a decimal, so that cents are exact. Every other type is
    correct by inference.
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
