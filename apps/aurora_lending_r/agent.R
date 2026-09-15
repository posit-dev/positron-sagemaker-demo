app_dir <- local({
  working_dir <- normalizePath(getwd())
  if (file.exists(file.path(working_dir, "agent.R"))) {
    working_dir
  } else {
    project_app <- file.path(working_dir, "apps", "aurora_lending_r")
    if (!file.exists(file.path(project_app, "agent.R"))) {
      stop("Run the app from its directory or the repository root.", call. = FALSE)
    }
    normalizePath(project_app)
  }
})

options(
  commons.context_cache = file.path(app_dir, "commons-cache"),
  # SageMaker Studio does not expose Landlock or unprivileged user namespaces.
  commons.allow_unsafe_fallback = TRUE
)

local_imagemagick <- path.expand("~/.local/commons-imagemagick")
if (dir.exists(local_imagemagick)) {
  local_imagemagick_lib <- file.path(local_imagemagick, "lib")
  existing_library_path <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(
    LD_LIBRARY_PATH = paste(
      c(local_imagemagick_lib, existing_library_path[nzchar(existing_library_path)]),
      collapse = ":"
    ),
    MAGICK_HOME = local_imagemagick
  )
}

aws_region <- function() Sys.getenv("POSIT_DEMO_REGION", "us-east-2")

aws_account_id <- function() {
  configured <- Sys.getenv("POSIT_DEMO_ACCOUNT_ID", "")
  if (nzchar(configured)) {
    return(configured)
  }

  paws::sts(config = list(region = aws_region()))$get_caller_identity()$Account
}

athena_bucket <- function() {
  configured <- Sys.getenv("POSIT_DEMO_BUCKET", "")
  if (nzchar(configured)) {
    return(configured)
  }

  sprintf(
    "sagemaker-posit-conf-2026-demo-%s-%s",
    aws_account_id(),
    aws_region()
  )
}

athena_connection_string <- function() {
  paste0(
    "Driver=", Sys.getenv("POSIT_DEMO_ATHENA_DRIVER", "Athena"), ";",
    "AwsRegion=", aws_region(), ";",
    "S3OutputLocation=s3://", athena_bucket(), "/athena-query-results/;",
    "AuthenticationType=", Sys.getenv("POSIT_DEMO_ATHENA_AUTH", "Default"), ";",
    "Workgroup=", Sys.getenv("POSIT_DEMO_WORKGROUP", "primary"), ";",
    "Schema=aurora_lending"
  )
}

connect_athena <- function() {
  DBI::dbConnect(
    odbc::odbc(),
    .connection_string = athena_connection_string()
  )
}

build_agent <- function(con = connect_athena()) {
  client <- ellmer::chat_aws_bedrock(
    model = Sys.getenv(
      "POSIT_DEMO_BEDROCK_MODEL",
      "us.anthropic.claude-sonnet-4-6"
    ),
    api = "converse",
    cache = "auto",
    params = ellmer::params(max_tokens = 4096),
    api_args = list(
      additionalModelRequestFields = list(
        thinking = list(type = "enabled", budget_tokens = 2000)
      )
    )
  )

  source <- commons::data_source(
    con,
    tables = c(
      "fct_loan_performance",
      "dim_borrower",
      "dim_loan_product"
    ),
    dictionary = file.path(app_dir, "dictionaries", "aurora_lending.yaml")
  )

  commons::commons(
    client = client,
    data_sources = list(aurora_lending = source),
    network = "none",
    semantic_layer = commons::semantic_layer(
      file.path(app_dir, "measures")
    ),
    context_layer = commons::context_layer(
      file.path(app_dir, "context", "portfolio-risk.md")
    )
  )
}
