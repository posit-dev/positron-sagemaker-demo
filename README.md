# Positron on Amazon SageMaker

Worked examples of a complete data science workflow inside Positron on Amazon
SageMaker. Each one reads governed data from Amazon Athena, explores it in the
IDE, trains and hosts a model on SageMaker, tracks the experiments in managed
MLflow, and publishes a report to Posit Connect. All of it happens in one place.

Everything here runs. All data is synthetic.

Use it to evaluate the stack, to lift a pattern into your own project, or to show
somebody how the pieces fit together. [PRESENTING.md](PRESENTING.md) has the
industry background and a walk-through script, if you are showing it to a room.

## The two examples

The repository holds two examples. They are alternatives, not two halves of one
story. Every script takes an explicit `--domain` so that nothing runs both by
accident.

| Domain | Company | Question | Athena database |
|---|---|---|---|
| `finance` | Aurora Lending Group | Which consumer loans will charge off? | `aurora_lending` |
| `lifesci` | Helix Therapeutics | Which trial subjects will leave the study? | `helix_trials` |

Both examples predict a yes-or-no outcome, for a reason. The modeling code and
the endpoint code are the same in each, so there is one pattern to learn.

## What you need

### In the AWS account

| Thing | Who makes it |
|---|---|
| S3 bucket, Glue databases, Glue tables | `load/load_athena.py`, automatically |
| SageMaker model, endpoint config, endpoint | `ml/train_and_deploy.py`, automatically |
| MLflow tracking server | `ml/mlflow_server.py create` |
| IAM execution role, with its policies | `setup/bootstrap_aws.py` |
| SageMaker domain | you, by hand. See below. |
| The Positron custom image, attached to that domain | you, by hand. See below. |

### On the machine you work from

- `uv` and `quarto`. The Positron SageMaker image contains both.
- AWS credentials. Inside the SageMaker image these come from the execution role, and there is nothing to configure.
- A Posit Connect server, if you want to publish the report.

## Start from a fresh AWS account

Run the bootstrap script from a workstation, with credentials that can write IAM.

```bash
uv sync
uv run python setup/bootstrap_aws.py --dry-run   # read what it will change
uv run python setup/bootstrap_aws.py
```

The script creates the execution role, gives it a trust policy for SageMaker,
attaches `AmazonSageMakerFullAccess`, renders the IAM template for your account
number, and then asks AWS to evaluate the result. Run it as often as you like. It
reports what already exists and changes nothing that is already correct.

Two things the script cannot do for you:

1. **Create the SageMaker domain.** Set its execution role to
   `sagemaker-demo-execution-role`. Use network mode `PublicInternetOnly`. If you
   must use `VpcOnly`, add VPC endpoints for `sagemaker.api`,
   `sagemaker.runtime`, `athena`, `glue`, and S3, or the demo cannot reach them.
2. **Build the Positron image and attach it to the domain.** Follow the Positron
   on Amazon SageMaker setup guide.

Then load the data and check the result:

```bash
uv run python data/generate.py    --domain lifesci
uv run python load/load_athena.py --domain lifesci
bash setup/verify-env.sh lifesci
```

### Two things to watch in a new account

**The region.** Amazon Athena and AWS Glue are available everywhere. A managed
MLflow tracking server is not. The bootstrap script tests for it and says so. If
your region has no managed MLflow, everything else still works and the
experiment tracking section is skipped.

**Lake Formation.** This demo relies on plain IAM to control access to Glue. That
works when Lake Formation uses `IAM_ALLOWED_PRINCIPALS` for new databases, which
is the default. The bootstrap script reports what your account uses. If it uses
something else, grant the execution role access to the two demo databases, or
Athena returns an access error that the IAM policy alone cannot explain.

## Install

```bash
uv sync
```

The SageMaker image contains few Python packages. This project brings its own
environment instead of depending on the image.

## Run an example

Select one domain. Then do the steps in order. The commands below use `finance`.
For the other example, use `--domain lifesci` and the `helix_trials` paths.

```bash
# 1. Make the synthetic data. The output is local parquet, and git ignores it.
uv run python data/generate.py --domain finance
uv run python data/validate.py --domain finance

# 2. Upload to S3 and register the tables in the AWS Glue Data Catalog.
uv run python load/load_athena.py   --domain finance
uv run python load/verify_athena.py --domain finance

# 3. If you want experiment tracking, start the MLflow tracking server.
uv run python ml/mlflow_server.py start

# 4. Train the model and host it on a SageMaker endpoint.
#    Each feature set becomes one MLflow run.
uv run python ml/train_and_deploy.py --domain finance
uv run python ml/smoke_test.py       --domain finance

# 5. Open analysis/walkthrough_finance.qmd in Positron. Run it cell by cell.

# 6. Render the report and publish it.
uv run quarto render reports/aurora_lending/portfolio_risk_review.qmd
bash setup/publish.sh finance

# 7. Remove the endpoint, and stop the MLflow server. Both stop billing.
uv run python ml/teardown.py --all
```

One command checks all of the above:

```bash
bash setup/verify-env.sh finance
```

CAUTION: Two resources bill by the hour. A real-time endpoint bills for every
hour that it exists, even when nothing calls it. An MLflow tracking server bills
$0.60 for every hour that it runs. Run `ml/teardown.py --all` when you finish.
That command removes the endpoints and stops the tracking server.

The tracking server is stopped and not removed, so every run stays and the next
start takes a few minutes. A stopped server bills only for storage, at $0.10 for
each GB in a month.

## Files

```
config.py                       AWS resource names, in one place
data/generate.py                makes the synthetic data
data/validate.py                checks the generated data
data/generators/                per-domain constants and table builders
data/data-dict.yaml             a description of every column
load/load_athena.py             uploads to S3 and registers Glue tables
load/verify_athena.py           makes sure the types survive the round trip
analysis/walkthrough_*.qmd      the interactive walkthroughs
ml/mlflow_server.py             creates, starts, stops, and removes the MLflow server
ml/tracking.py                  logs runs to managed MLflow
ml/train_and_deploy.py          trains the model and hosts it on SageMaker
ml/entrypoint/inference.py      the serving entrypoint, which uses only numpy
ml/smoke_test.py                makes sure a live endpoint scores correctly
ml/teardown.py                  removes endpoints so that they stop billing
reports/                        the Quarto reports for Posit Connect
reports/requirements.txt        the packages needed to render, and no more
tests/test_report_config.py     makes sure the reports agree with config.py
PRESENTING.md                   background, and a script for showing this to a room
LICENSE                         MIT
iam/                            IAM policy templates and a script to apply them
setup/bootstrap_aws.py          prepares a fresh AWS account
setup/verify-env.sh             checks the environment in one command
setup/publish.sh                renders and publishes to Connect
```

The walkthroughs are Quarto documents, not notebooks. The SageMaker image
removes every Jupyter kernelspec, so it cannot run a notebook.

Positron runs a Quarto code cell straight into the console, so a walkthrough
still fills the Variables pane, the Plots pane, and the Data Explorer. You also
get the explanation next to the code, and you can render the whole file:

```bash
uv run quarto render analysis/walkthrough_finance.qmd
```

If the endpoint is absent, the model sections print what to deploy and the rest
of the file still runs.

## Data

`data/generate.py` makes all of the data from a fixed seed, so every run gives
the same result. The files go to `data/synthetic-<database>/`, and git ignores
them. If the data is absent, each script tells you which command to run.

The row counts are small for a reason. There are 50,000 loans and 1,200 trial
subjects. Athena answers in about one second, so you are never waiting on a
query.

## No account number in this repository

This repository is public, so it contains no AWS account number.

- `config.py` reads the account from AWS STS and builds the bucket name from it.
- The reports do the same, because they cannot import `config.py`.
- The IAM files are templates. They hold `ACCOUNT_ID` and `REGION` placeholders.

A fork works without an edit. To use a different account or region, set these
environment variables:

| Variable | Effect |
|---|---|
| `POSIT_DEMO_REGION` | the AWS region. The default is `us-east-2`. |
| `POSIT_DEMO_ACCOUNT_ID` | the account number. This avoids the STS call. |
| `POSIT_DEMO_BUCKET` | the S3 bucket name, in full. |
| `POSIT_DEMO_WORKGROUP` | the Athena workgroup. The default is `primary`. |
| `POSIT_DEMO_INSTANCE_TYPE` | the endpoint instance type. The default is `ml.m5.large`. |
| `POSIT_DEMO_MLFLOW_SERVER` | the MLflow tracking server name. |
| `POSIT_DEMO_MLFLOW_SIZE` | the tracking server size. The default is `Small`. |

## IAM

There are two policy templates.

**`iam/sagemaker-demo-athena-access.template.json`** goes on the SageMaker
execution role. `AmazonSageMakerFullAccess` already grants S3 access, because the
bucket name contains `sagemaker`, and it grants `sagemaker:InvokeEndpoint`. It
grants no Athena action, and only `glue:GetTable*`. This template adds the Athena
and Glue reads.

```bash
bash iam/apply-policy.sh sagemaker-demo-athena-access sagemaker-demo-execution-role
```

**`iam/connect-reader-policy.template.json`** is for the principal that Posit
Connect uses. The published report renders against live Athena, so Connect needs
these permissions. The policy is read-only. It covers the two demo databases, the
demo bucket, and `sagemaker:InvokeEndpoint` on the two demo endpoints only.

Note that `athena:ListWorkGroups`, `athena:ListDataCatalogs`, and
`athena:ListEngineVersions` are account-level actions. They do not accept a
resource ARN. For this reason they have their own statement, with `Resource: "*"`.
If you scope them to a workgroup ARN, AWS denies them.

### Experiment tracking

The policy also grants `sagemaker-mlflow:*` on the demo tracking server, and it
grants the SageMaker actions that describe, start, and stop that server.
`AmazonSageMakerFullAccess` grants no `sagemaker-mlflow` action at all, so these
statements are necessary.

The tracking server writes artifacts to `s3://<bucket>/mlflow/` under the Studio
execution role. That role already reaches the bucket, because the bucket name
holds `sagemaker`, so no new role is necessary.

### The policy is read-only, so queries must not use CTAS

`awswrangler.athena.read_sql_query` uses `ctas_approach=True` by default. That
option makes a temporary Glue table for every query, then removes it. It needs
`glue:CreateTable` and `glue:DeleteTable`. This policy does not grant them.

Every query in this repository passes `ctas_approach=False`. If you add a query
and forget this, the query works for a user with broad permissions and fails for
the demo role. A policy simulation shows the difference:

```
allowed       glue:GetTable       aurora_lending/fct_loan_performance
implicitDeny  glue:CreateTable    aurora_lending/tmp_ctas
```

## Experiment tracking

Amazon SageMaker hosts a managed MLflow tracking server. Training logs one run
for each feature set, so the MLflow UI shows a comparison and not a single row.

```bash
uv run python ml/mlflow_server.py create   # once, about 22 minutes
uv run python ml/mlflow_server.py start    # when you want tracking
uv run python ml/mlflow_server.py url      # open the UI
uv run python ml/mlflow_server.py stop     # when you are done
```

Creating a server took 22 minutes when this was written. Stopping one took more
than 10 minutes, so start the stop and do not wait for it.

Tracking is optional. If the server is stopped or absent, training prints the
reason and continues. The check that decides this uses `boto3` and answers in
about one second. The MLflow client itself needs more than four minutes to
report that a server is absent, which is too slow for a live session.

## Publish to Connect

Each report is a self-contained document. The `.qmd` file declares the settings
it needs instead of importing `config.py`, because the Connect bundle does not
contain `config.py`. `tests/test_report_config.py` fails if the two disagree, and
`setup/publish.sh` runs that test before every deploy.

Connect renders the report against live Athena, so Connect needs AWS credentials
as content variables:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` | credentials for the principal that holds the Connect reader policy |
| `AWS_DEFAULT_REGION` | the region of the demo data |

The `POSIT_DEMO_*` variables above also work on Connect.

If Connect has no AWS access, publish the rendered file instead:

```bash
uv run quarto publish connect reports/aurora_lending/portfolio_risk_review.qmd
```

The report then cannot refresh itself, but nothing else changes. In both cases,
if the endpoint is absent, the model section becomes a short note and the rest of
the report still renders. Published content survives the removal of an endpoint.

## Branding

`_brand.yml` holds the Posit color palette. Both reports use it. Point it at
another brand file to change the appearance.

## Use these parts elsewhere

The pieces work on their own. Three are useful well beyond these examples:

- the Athena access pattern, which needs no credentials inside SageMaker
- the JSON scorecard entrypoint, which removes any scikit-learn version link between training and serving
- the report behavior when an endpoint is absent

## Important Disclaimer

**This project contains synthetic data and analysis created for demonstration
purposes only.**

All data, insights, business scenarios, and analytics presented in this
demonstration project have been artificially generated using AI. The data does
not represent actual business information, performance metrics, customer data,
or operational statistics.

### Key Points:

- **Synthetic Data**: All datasets are computer-generated and designed to
  illustrate analytical capabilities
- **Illustrative Analysis**: Insights and recommendations are examples of the
  types of analysis possible with Posit tools
- **No Actual Business Data**: No real business information or data was used or
  accessed in creating this demonstration
- **Educational Purpose**: This project serves as a technical demonstration of
  data science workflows and reporting capabilities
- **AI-Generated Content**: Analysis, commentary, and business scenarios were
  created by AI for illustration purposes
- **No Real-World Implications**: The scenarios and insights presented should
  not be interpreted as actual business advice or strategies

This demonstration showcases how Posit's commercial and open-source tools can be
applied to the financial services and life sciences industry. The synthetic data
and analysis provide a foundation for understanding the potential value of
implementing similar analytical workflows with actual business data.

For questions about adapting these techniques to your real business scenarios,
please contact your Posit representative.
---

*This demonstration was created using Posit's commercial data science tools and
open-source packages. All synthetic data and analysis are provided for
evaluation purposes only.*
