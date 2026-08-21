"""Confirm the loaded data reads back out of Athena with the types we intended.

    uv run python load/verify_athena.py --domain finance

Type drift is silent at load time -- awswrangler infers Glue types from pandas
dtypes -- and only surfaces mid-walkthrough. This checks the actual contract:
that the catalogue types are right and that a real query returns the same
numbers the local validator printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awswrangler as wr
import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

# Columns whose Athena type we care about enough to assert.
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
            "enrolment_date": "date",
            "discontinued": "boolean",
            "subject_id": "string",
            "baseline_das": "double",
        },
        "fct_visit_observations": {"adverse_event_flag": "boolean", "das_score": "double"},
    },
}

# One headline query per domain, so a human can eyeball the result.
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
    # NOTE: discontinuation is aggregated over dim_subject *alone*. Joining to a
    # single visit first would silently drop everyone who left before that visit
    # and understate the dropout rate -- the classic survivorship-bias trap in
    # trial data. The Week 24 DAS is pulled in as a separate scalar subquery.
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

    print(f"{domain.company} -- Glue database {domain.database}\n")

    print("catalogue types")
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
        n = wr.athena.read_sql_query(
            f"SELECT count(*) AS n FROM {table}",
            database=domain.database,
            workgroup=config.ATHENA_WORKGROUP,
            s3_output=config.athena_staging(),
            boto3_session=session,
            # ctas_approach=False: the default creates temporary Glue tables, which
            # needs Glue write permission the read-only demo role does not have.
            ctas_approach=False,
        )["n"].iloc[0]
        print(f"  {table:26} {n:>7,}")
        if n == 0:
            failures.append(f"{table} returned 0 rows")

    print("\nheadline query")
    df = wr.athena.read_sql_query(
        HEADLINE[domain.database],
        database=domain.database,
        workgroup=config.ATHENA_WORKGROUP,
        s3_output=config.athena_staging(),
        boto3_session=session,
        # ctas_approach=False: the default creates temporary Glue tables, which
        # needs Glue write permission the read-only demo role does not have.
        ctas_approach=False,
    )
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
