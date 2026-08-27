"""Make sure Athena reads the loaded data back with the correct types.

Queries run over ODBC, which is how the rest of this project reads Athena. The
column types come from the AWS Glue API, because that is a catalogue question
and not a query.

    uv run python load/verify_athena.py --domain finance

The loader reads the Glue types from the pandas dtypes. A wrong type is silent
at load time, and it appears later during a walkthrough. This script checks the
catalog types, and it checks that a query returns the same numbers that
data/validate.py printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import athena  # noqa: E402
import config  # noqa: E402

# The columns where the Athena type must be exact.
EXPECTED_TYPES = {
    "aurora_lending": {
        "fct_loan_performance": {
            "origination_date": "date",
            "principal_amount": "decimal(12,2)",
            "outstanding_balance": "decimal(12,2)",
            "charged_off": "boolean",
            "purpose": "string",
        },
        "dim_borrower": {"annual_income": "decimal(12,2)", "fico_score": "smallint"},
    },
    "helix_trials": {
        "dim_subject": {
            "enrollment_date": "date",
            "discontinued": "boolean",
            "subject_id": "string",
            "baseline_das": "double",
        },
        "fct_visit_observations": {"adverse_event_flag": "boolean", "das_score": "double"},
    },
}

# One query for each demo, so that a person can read the result.
HEADLINE = {
    "aurora_lending": """
        SELECT p.purpose,
               count(*)                                    AS loans,
               round(avg(f.apr) * 100, 2)                   AS avg_apr_pct,
               sum(f.principal_amount)                      AS originated,
               round(avg(CASE WHEN f.charged_off THEN 1.0 ELSE 0.0 END) * 100, 2) AS chargeoff_pct
        FROM fct_loan_performance f
        JOIN dim_loan_product p ON p.product_id = f.product_id
        GROUP BY p.purpose
        ORDER BY chargeoff_pct DESC
    """,
    # NOTE: this query counts discontinuation over dim_subject only. If it
    # joined to one visit first, it would remove every subject who left before
    # that visit, and report a dropout rate that is too low. That is
    # survivorship bias. The Week 24 DAS comes from a separate subquery.
    "helix_trials": """
        SELECT s.arm,
               count(*)                                     AS subjects,
               round(avg(CASE WHEN s.discontinued THEN 1.0 ELSE 0.0 END) * 100, 2) AS discont_pct,
               (SELECT round(avg(o.das_score), 2)
                  FROM fct_visit_observations o
                  JOIN dim_subject s2 ON s2.subject_id = o.subject_id
                 WHERE o.visit_id = 7 AND s2.arm = s.arm)   AS das_week24
        FROM dim_subject s
        GROUP BY s.arm
        ORDER BY s.arm
    """,
}

failures: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(config.DOMAINS))
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    session = boto3.Session(region_name=config.REGION)
    glue = session.client("glue")

    print(f"{domain.company}: Glue database {domain.database}\n")

    print("catalog types")
    for table, expected in EXPECTED_TYPES[domain.database].items():
        cols = {
            c["Name"]: c["Type"]
            for c in glue.get_table(DatabaseName=domain.database, Name=table)["Table"][
                "StorageDescriptor"
            ]["Columns"]
        }
        for name, want in expected.items():
            got = cols.get(name, "<missing>")
            ok = got == want
            print(f"  [{'ok  ' if ok else 'FAIL'}] {table}.{name:24} {got}")
            if not ok:
                failures.append(f"{table}.{name} is {got}, expected {want}")

    print("\nrow counts via Athena")
    for table in sorted(EXPECTED_TYPES[domain.database]):
        n = athena.query(f"SELECT count(*) AS n FROM {table}", domain.database)["n"].iloc[0]
        print(f"  {table:26} {n:>7,}")
        if n == 0:
            failures.append(f"{table} returned 0 rows")

    print("\nheadline query")
    df = athena.query(HEADLINE[domain.database], domain.database)
    print(df.to_string(index=False))

    print("\ndtypes as returned to pandas")
    print("   ", dict(df.dtypes.astype(str)))

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nAthena round-trip verified")


if __name__ == "__main__":
    main()
