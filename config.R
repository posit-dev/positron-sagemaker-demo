# AWS resource names for the demo, in one place.
#
# This is the R port of config.py. This repository holds two demos:
# Aurora Lending Group (finance) and Helix Therapeutics (lifesci). They are
# alternatives for different sessions, so pick one. The two files must stay
# in agreement -- if you change a default here, change it in config.py too.

athena_region <- function() Sys.getenv("POSIT_DEMO_REGION", "us-east-2")

# The account comes from AWS STS, so no account number is written into this
# public repository, and a fork works with no edit. To avoid the STS call,
# set POSIT_DEMO_ACCOUNT_ID.
account_id <- function() {
  from_env <- Sys.getenv("POSIT_DEMO_ACCOUNT_ID", "")
  if (nzchar(from_env)) {
    return(from_env)
  }
  paws::sts(config = list(region = athena_region()))$get_caller_identity()$Account
}

bucket_prefix <- "sagemaker-posit-conf-2026-demo"

# The name starts with "sagemaker" for a reason: AmazonSageMakerFullAccess
# grants S3 object access on arn:aws:s3:::*sagemaker*, so the Studio execution
# role needs no S3 policy for this bucket.
bucket <- function() {
  override <- Sys.getenv("POSIT_DEMO_BUCKET", "")
  if (nzchar(override)) {
    return(override)
  }
  sprintf("%s-%s-%s", bucket_prefix, account_id(), athena_region())
}

# Workgroup "primary" has EnforceWorkGroupConfiguration=false and no
# OutputLocation, so every client must pass staging explicitly.
athena_workgroup <- function() Sys.getenv("POSIT_DEMO_WORKGROUP", "primary")

athena_staging <- function() sprintf("s3://%s/athena-query-results/", bucket())

# --- Athena over ODBC --------------------------------------------------------
# The Positron SageMaker image ships the Posit Professional Drivers,
# registered with unixODBC. Two values differ between drivers, so both read an
# environment variable. The defaults are the ones the image uses.
#
#   driver name  The image registers the Posit driver as "Athena".
#   auth type    The Posit driver takes "Default". The Amazon Athena ODBC 2.x
#                driver, which is what a workstation normally has, calls the
#                same thing "Default Credentials".
#
# Do not use "Instance Profile". That selects the EC2 metadata path, which is
# not how a Studio app receives its credentials.
athena_odbc_driver <- function() Sys.getenv("POSIT_DEMO_ATHENA_DRIVER", "Athena")
athena_odbc_auth <- function() Sys.getenv("POSIT_DEMO_ATHENA_AUTH", "Default")

athena_odbc_string <- function(database) {
  paste0(
    "Driver=", athena_odbc_driver(), ";",
    "AwsRegion=", athena_region(), ";",
    "S3OutputLocation=", athena_staging(), ";",
    "AuthenticationType=", athena_odbc_auth(), ";",
    "Workgroup=", athena_workgroup(), ";",
    "Schema=", database
  )
}

# --- Managed MLflow -----------------------------------------------------------
mlflow_server_name <- function() {
  Sys.getenv("POSIT_DEMO_MLFLOW_SERVER", "posit-conf-2026-demo")
}

# --- The values that differ between the two demos ----------------------------
domains <- list(
  finance = list(
    key = "finance",
    company = "Aurora Lending Group",
    industry = "financial services",
    database = "aurora_lending",
    fact_table = "fct_loan_performance",
    target = "charged_off",
    endpoint = "aurora-lending-risk",
    experiment = "aurora-lending-charge-off",
    report = "reports/aurora_lending/portfolio_risk_review.qmd"
  ),
  lifesci = list(
    key = "lifesci",
    company = "Helix Therapeutics",
    industry = "life sciences",
    database = "helix_trials",
    fact_table = "fct_visit_observations",
    target = "discontinued",
    endpoint = "helix-dropout-risk",
    experiment = "helix-trials-dropout",
    report = "reports/helix_trials/enrollment_safety_review.qmd"
  )
)

get_domain <- function(key) {
  if (!key %in% names(domains)) {
    stop(sprintf("Unknown domain '%s'. Choose one of: %s", key,
                 paste(names(domains), collapse = ", ")))
  }
  domains[[key]]
}
