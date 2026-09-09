# Read Amazon Athena through ODBC.
#
# This is the R port of athena.py. The Positron SageMaker image ships the
# Posit Professional Drivers and registers them with unixODBC, so the same
# connection string that pyodbc uses also works with the odbc package -- the
# reason the project prefers ODBC for reads over a Python-only client.
#
# Writes are a different matter. Creating tables and registering them in the
# AWS Glue Data Catalog has no ODBC equivalent, so load/load_athena.py uses
# awswrangler; there is no R port of that script.
#
#   source("config.R")
#   source("athena.R")
#   frame <- athena_query("SELECT count(*) AS n FROM dim_subject", "helix_trials")

library(DBI)

#' Open an ODBC connection to one Glue database.
athena_connect <- function(database) {
  available <- tryCatch(odbc::odbcListDrivers()$name, error = function(e) character())
  if (!(athena_odbc_driver() %in% available)) {
    stop(
      "unixODBC has no driver named '", athena_odbc_driver(), "'.\n",
      "  drivers it can see: ", paste(available, collapse = ", "), "\n\n",
      "  In the Positron SageMaker image the Posit Professional Drivers\n",
      "  register this name already.\n\n",
      "  On a workstation, install the Amazon Athena ODBC 2.x driver, then\n",
      "  point these at it:\n",
      "    Sys.setenv(POSIT_DEMO_ATHENA_DRIVER = 'Amazon Athena ODBC Driver')\n",
      "    Sys.setenv(POSIT_DEMO_ATHENA_AUTH = 'Default Credentials')",
      call. = FALSE
    )
  }
  DBI::dbConnect(odbc::odbc(), .connection_string = athena_odbc_string(database))
}

#' Run one query and return a data frame, closing the connection afterward.
athena_query <- function(sql, database) {
  con <- athena_connect(database)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  DBI::dbGetQuery(con, sql)
}

#' The tables in one database, read through Athena rather than the Glue API.
athena_tables <- function(database) {
  athena_query(
    paste0(
      "SELECT table_name, table_type ",
      "FROM information_schema.tables ",
      "WHERE table_schema = '", database, "' ",
      "ORDER BY table_name"
    ),
    database
  )
}
