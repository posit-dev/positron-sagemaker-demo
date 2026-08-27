"""Read Amazon Athena through ODBC.

The Positron SageMaker image ships the Posit Professional Drivers and preinstalls
pyodbc, so every read in this project uses that toolchain. The same connection
string works from R with the `odbc` package, which is the reason to prefer it
over a Python-only client.

Writes are a different matter. Creating tables and registering them in the AWS
Glue Data Catalog has no ODBC equivalent, so load/load_athena.py uses awswrangler.

    from athena import query
    frame = query("SELECT count(*) AS n FROM dim_subject", "helix_trials")
"""

from __future__ import annotations

import warnings

import pandas as pd

import config


class DriverNotFound(RuntimeError):
    """Raised when unixODBC cannot see an Athena driver."""


def _fail_without_driver(available: list[str]) -> "DriverNotFound":
    return DriverNotFound(
        f"unixODBC has no driver named {config.ATHENA_ODBC_DRIVER!r}.\n"
        f"  drivers it can see: {', '.join(available) or 'none'}\n"
        "\n"
        "  In the Positron SageMaker image the Posit Professional Drivers\n"
        "  register this name already.\n"
        "\n"
        "  On a workstation, install the Amazon Athena ODBC 2.x driver, then\n"
        "  point these at it:\n"
        "    export POSIT_DEMO_ATHENA_DRIVER='Amazon Athena ODBC Driver'\n"
        "    export POSIT_DEMO_ATHENA_AUTH='Default Credentials'"
    )


def connect(database: str):
    """Open an ODBC connection to one Glue database."""
    import pyodbc

    if config.ATHENA_ODBC_DRIVER not in pyodbc.drivers():
        raise _fail_without_driver(pyodbc.drivers())
    return pyodbc.connect(config.athena_odbc_string(database), autocommit=True)


def query(sql: str, database: str) -> pd.DataFrame:
    """Run one query and return a data frame."""
    with connect(database) as connection:
        with warnings.catch_warnings():
            # pandas prefers a SQLAlchemy connectable and says so for any other
            # DBAPI connection. A DBAPI connection is what pyodbc gives, and it
            # reads correctly.
            warnings.filterwarnings("ignore", message=".*DBAPI2.*")
            return pd.read_sql(sql, connection)


def tables(database: str) -> pd.DataFrame:
    """The tables in one database, read through Athena rather than the Glue API."""
    return query(
        "SELECT table_name, table_type "
        "FROM information_schema.tables "
        f"WHERE table_schema = '{database}' "
        "ORDER BY table_name",
        database,
    )
