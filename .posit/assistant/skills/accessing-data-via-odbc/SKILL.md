---
name: accessing-data-via-odbc
description: >-
  Connects to and queries external data sources (data warehouses, databases,
  data lakes) through ODBC using R (DBI + odbc) or Python (pyodbc/turbodbc,
  optionally SQLAlchemy or Polars). Use whenever Assistant needs to actually
  explore, profile, or query data that lives in an external store — Athena,
  Redshift, Snowflake, Databricks, SQL Server, Postgres, or any other
  ODBC-accessible source — from an R or Python console. CLI tools and SDKs
  (e.g. AWS CLI, boto3, the snow CLI) remain appropriate for discovery —
  listing catalogs, databases, tables, schemas, or metadata. Other skills
  (e.g. querying-data-lake, snowflake) may own that discovery step and supply
  connection details, catalog structure, or SQL patterns for a specific
  platform; once the task shifts to actually reading, sampling, or analyzing
  rows of data, this skill governs how that happens: through ODBC in the R
  or Python session, not by continuing with CLI/SDK query mechanisms like
  Athena start-query-execution polling. Triggers on phrases like: connect to
  the database, ODBC connection, DSN, query the warehouse, pull this table
  into R/Python, read_odbc, dbConnect, pyodbc.connect.
metadata:
  version: "1"
---

# Accessing Data via ODBC

Meta-skill for connecting to and querying external data sources from the R or Python console. When any other skill describes *what* to query (table names, catalogs, SQL patterns, cost controls), this skill governs *how* the connection and query execution happen: through an ODBC driver in the active R or Python session, using the current language session's tooling — not CLI commands, REST calls, or non-ODBC SDKs.

## Discovery vs. querying — draw this line explicitly

Two different jobs are easy to conflate; keep them separate:

- **Discovery/metadata (CLI or SDK is fine)**: finding out *what* data exists — listing catalogs, databases, tables, schemas, columns, partitions, workgroups, or checking table size/ownership/permissions. Tools like the AWS CLI, `boto3`, Athena/Glue APIs, `snow` CLI, etc. are appropriate here, and other skills (e.g. `querying-data-lake`, `finding-data-lake-assets`, `snowflake`) may already own this step.
- **Working with the data (ODBC only)**: once the goal shifts to actually reading rows, profiling values, sampling, joining, aggregating, or otherwise pulling data into R/Python for analysis — switch to an ODBC connection (DBI/odbc in R, pyodbc in Python) rather than continuing with the CLI/SDK path (e.g. Athena `start-query-execution` + polling, `boto3` result parsing).

Rule of thumb: if the next step is "tell me about this data" (names, schema, counts, metadata), CLI/SDK discovery is fine. If the next step is "show me / give me the data" (rows, values, a data frame to analyze), use ODBC.

## When this skill applies

- The user asks to explore, profile, sample, or analyze the *content* of data that lives outside the local session (a database, warehouse, or lake) — i.e., anything beyond metadata.
- Another skill has identified a target table/catalog/connection via discovery tooling but the next step is to load rows into R/Python — use this skill to finish the job via ODBC instead of continuing with that skill's CLI/SDK query mechanism.
- The user mentions ODBC, a DSN, a connection string, `dbConnect`, `pyodbc`, or similar.

Prefer the active foreground language session (R or Python) unless the user specifies otherwise. If neither session has the required driver installed, say so and ask before installing anything.

## Discover before connecting

1. Check for an existing connection already open in the session (list session variables/objects for a `DBIConnection` or `pyodbc.Connection`/SQLAlchemy engine) — reuse it rather than opening a duplicate.
2. Check for configured DSNs before asking the user to supply a full connection string:
   - R: `odbc::odbcListDataSources()`
   - Python: `pyodbc.dataSources()`
3. Check for connection details already stored as environment variables, a `.odbc.ini`/`odbcinst.ini`, or a Positron/Workbench-managed connection (visible in the Connections pane). Prefer these over asking the user to retype credentials.
4. Never hardcode credentials in code. Use environment variables, a credentials manager, or Workbench-managed credentials, and say which one you used.

## R workflow (DBI + odbc)

```r
library(DBI)

con <- dbConnect(
  odbc::odbc(),
  dsn = "my_dsn"                 # or drop dsn= and pass driver/server/etc. directly
)

dbListTables(con)
df <- dbGetQuery(con, "SELECT * FROM schema.table LIMIT 100")

dbDisconnect(con)
```

- Use `dbGetQuery()` for read-only exploration; use `dbSendQuery()`/`dbFetch()` only when you need paged results.
- Always disconnect (`dbDisconnect(con)`) when a one-off exploration is finished; leave long-lived connections open only if the user is clearly working interactively across multiple turns.

## Python workflow (pyodbc)

```python
import pyodbc
import pandas as pd

conn = pyodbc.connect("DSN=my_dsn")  # or a full connection string

tables = pd.read_sql(
    "SELECT table_name FROM information_schema.tables", conn
)
df = pd.read_sql("SELECT * FROM schema.table LIMIT 100", conn)

conn.close()
```

- If the project uses Polars, prefer `pl.read_database(query, connection=conn)` over `pd.read_sql()` for consistency — match whichever data-manipulation package the project already uses (see general Polars/pandas guidance).
- `sqlalchemy` with the `pyodbc` dialect is an acceptable alternative when the user or project already depends on SQLAlchemy; don't introduce it otherwise.
- Close the connection when the exploration is done, same as in R.

## Building the connection string

If no DSN exists, connection strings vary by driver/platform (Athena, Redshift, Snowflake, Databricks, SQL Server, Postgres). See [references/connection-strings.md](references/connection-strings.md) for examples. Ask the user for the platform and any required parameters (host, port, warehouse/catalog, auth method) you can't infer from context before constructing one.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Data source name not found` | DSN missing from `odbc.ini`/`odbcinst.ini` or wrong casing | List DSNs with `odbcListDataSources()`/`pyodbc.dataSources()`; confirm exact name with the user |
| `Driver not found` / `libodbc.so` errors | ODBC driver manager or platform driver not installed | Report the missing driver; ask before installing (system package, not `pip install`/`uv add`) |
| Query hangs or times out | Long-running query against a large remote table | Suggest adding `LIMIT`/`TOP` for exploration, or ask before running the full query |
| Connection works but no tables listed | Wrong schema/catalog context | Use `dbGetQuery(con, "SELECT current_schema()")` (or platform equivalent) to check context |
| Credentials rejected | Expired token or wrong auth method for the platform | Check whether the platform uses token-based auth (e.g., Snowflake SSO, Workbench-managed credentials) rather than a static password |

## Additional Resources

- [Connection string examples by platform](references/connection-strings.md)
- [R odbc package documentation](https://solutions.posit.co/connections/db/r-packages/odbc/)
- [Python pyodbc documentation](https://github.com/mkleehammer/pyodbc/wiki)
