# Connection String Examples by Platform

Use a configured DSN whenever one exists — check `odbcListDataSources()` (R) or `pyodbc.dataSources()` (Python) first. Build a raw connection string only when no DSN is available. Ask the user for any placeholder value you can't infer.

## Contents

- Amazon Athena
- Amazon Redshift
- Snowflake
- Databricks
- SQL Server
- PostgreSQL

## Amazon Athena

Requires the Athena ODBC driver (Simba). Needs an S3 staging directory for query results.

```
Driver=Simba Athena ODBC Driver;
AwsRegion=<region>;
Schema=<database>;
S3OutputLocation=s3://<bucket>/<prefix>/;
AuthenticationType=<Default Credentials Provider Chain | IAM Credentials | ...>;
Workgroup=<workgroup>;
```

## Amazon Redshift

Requires the Amazon Redshift ODBC driver.

```
Driver=Amazon Redshift ODBC Driver;
Server=<cluster-or-workgroup-endpoint>;
Port=5439;
Database=<database>;
UID=<user>;
PWD=<password>;
```

For Redshift Serverless, use the workgroup endpoint as `Server`. Prefer IAM/Workbench-managed credentials over a static `PWD` when available.

## Snowflake

Requires the Snowflake ODBC driver.

```
Driver=SnowflakeDSIIDriver;
Server=<account>.snowflakecomputing.com;
Database=<database>;
Schema=<schema>;
Warehouse=<warehouse>;
UID=<user>;
Authenticator=<snowflake | externalbrowser | oauth>;
```

Prefer `externalbrowser` or `oauth` authenticators over a static password when SSO is configured. See the `snowflake` skill for querying semantic views once connected.

## Databricks

Requires the Databricks ODBC/Simba Spark driver.

```
Driver=Simba Spark ODBC Driver;
Host=<workspace-hostname>;
Port=443;
HTTPPath=<http-path-from-cluster-or-warehouse>;
AuthMech=3;
UID=token;
PWD=<personal-access-token>;
```

Prefer a Databricks personal access token stored as an environment variable, not inline in code.

## SQL Server

```
Driver=ODBC Driver 18 for SQL Server;
Server=<host>,<port>;
Database=<database>;
UID=<user>;
PWD=<password>;
Encrypt=yes;
```

## PostgreSQL

```
Driver=PostgreSQL Unicode;
Server=<host>;
Port=5432;
Database=<database>;
UID=<user>;
PWD=<password>;
```
