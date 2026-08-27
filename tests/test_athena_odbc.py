"""Prove that Athena answers over ODBC, with the driver the image ships.

    uv run python tests/test_athena_odbc.py

This test needs an Athena ODBC driver. The Positron SageMaker image registers
the Posit Professional Drivers as "Athena" and authenticates with the execution
role, which is what config.py expects by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


def main() -> int:
    import pyodbc

    print(f"driver   {config.ATHENA_ODBC_DRIVER}")
    print(f"auth     {config.ATHENA_ODBC_AUTH}")
    print(f"visible  {pyodbc.drivers()}")

    if config.ATHENA_ODBC_DRIVER not in pyodbc.drivers():
        print(f"\nFAIL: unixODBC cannot see a driver named "
              f"{config.ATHENA_ODBC_DRIVER!r}")
        return 1

    failures = 0
    for domain in config.DOMAINS.values():
        connection_string = config.athena_odbc_string(domain.database)
        try:
            with pyodbc.connect(connection_string, autocommit=True) as connection:
                cursor = connection.cursor()
                count = cursor.execute(
                    f"SELECT count(*) FROM {domain.fact_table}").fetchone()[0]
                print(f"  [ok  ] {domain.database}.{domain.fact_table}  {count:,} rows")

                # Types matter as much as the row count. Money must stay exact
                # and a date must not arrive as a string.
                if domain.key == "finance":
                    row = cursor.execute(
                        "SELECT principal_amount, origination_date, charged_off "
                        "FROM fct_loan_performance LIMIT 1").fetchone()
                    print(f"  [ok  ] types  {[type(v).__name__ for v in row]}  {tuple(row)}")
        except Exception as err:  # noqa: BLE001
            print(f"  [FAIL] {domain.database}: {type(err).__name__}: {err}")
            failures += 1

    # athena.query() reads through pandas, not a raw cursor, so exercise that
    # path as well. pandas prefers a SQLAlchemy connectable and warns for any
    # other DBAPI connection.
    import athena

    for domain in config.DOMAINS.values():
        try:
            frame = athena.query(
                f"SELECT * FROM {domain.fact_table} LIMIT 5", domain.database)
            print(f"  [ok  ] pandas {domain.database}  {frame.shape[0]} rows x "
                  f"{frame.shape[1]} cols")
            print(f"         dtypes {dict(list(frame.dtypes.astype(str).items())[:4])}")
        except Exception as err:  # noqa: BLE001
            print(f"  [FAIL] pandas {domain.database}: {type(err).__name__}: {err}")
            failures += 1

    try:
        listing = athena.tables("helix_trials")
        print(f"  [ok  ] information_schema listing  {len(listing)} tables")
    except Exception as err:  # noqa: BLE001
        print(f"  [FAIL] table listing: {type(err).__name__}: {err}")
        failures += 1

    print("\nAthena answers over ODBC" if not failures else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
