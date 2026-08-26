# Positron on Amazon SageMaker

Worked examples of a complete data science workflow inside Positron on Amazon
SageMaker. Each one reads governed data from Amazon Athena and explores it in the
IDE. It then trains a model, hosts it on SageMaker, and tracks the experiments in
managed MLflow. Last, it publishes a report to Posit Connect. All of it happens
in one place.

Everything here runs. All data is synthetic.

https://github.com/user-attachments/assets/2679c3ee-a793-4a94-98db-0e6c64dde5c8

## The two examples

They are alternatives, not two halves of one story. Every script takes an
explicit `--domain` so that nothing runs both by accident.

| Domain | Company | Question | Athena database |
|---|---|---|---|
| `finance` | Aurora Lending Group | Which consumer loans will charge off? | `aurora_lending` |
| `lifesci` | Helix Therapeutics | Which trial subjects will leave the study? | `helix_trials` |

Both predict a yes-or-no outcome, for a reason. The modeling code and the
endpoint code are the same in each, so there is one pattern to learn.

## What you need

| Thing | Who makes it |
|---|---|
| S3 bucket, Glue databases and tables | `load/load_athena.py` |
| SageMaker model, endpoint config, endpoint | `ml/train_and_deploy.py` |
| MLflow tracking server | `ml/mlflow_server.py create` |
| IAM execution role and its policies | `setup/bootstrap_aws.py` |
| SageMaker domain | you |
| The Positron custom image, attached to that domain | you |

On the machine you work from you need `uv` and `quarto`. The Positron SageMaker
image contains both. Inside that image the AWS credentials come from the
execution role, and there is nothing to configure.

## Start from a fresh AWS account

Run this from a workstation, with credentials that can write IAM.

```bash
uv sync
uv run python setup/bootstrap_aws.py --dry-run   # read what it will change
uv run python setup/bootstrap_aws.py
```

The script creates the execution role, attaches `AmazonSageMakerFullAccess`,
renders the IAM template for your account number, and asks AWS to evaluate the
result. Run it as often as you like. It changes nothing that is already correct.

Two steps stay with you:

1. **Create the SageMaker domain.** Set its execution role to
   `sagemaker-demo-execution-role`. Use network mode `PublicInternetOnly`. If you
   must use `VpcOnly`, add VPC endpoints for `sagemaker.api`,
   `sagemaker.runtime`, `athena`, `glue`, and S3.
2. **Build the Positron image and attach it to the domain.** Follow the Positron
   on Amazon SageMaker setup guide.

Two things differ between accounts, and the script reports both. Amazon Athena
and AWS Glue are available everywhere, but a managed MLflow server is not.
Everything except experiment tracking works without it. Separately, this project
relies on plain IAM to control Glue access, which works while Lake Formation uses
`IAM_ALLOWED_PRINCIPALS` for new databases. That is the default. If your account
differs, grant the execution role access to the two databases.

## Run an example

Select one domain, then do the steps in order. These commands use `finance`. For
the other example, use `--domain lifesci` and the `helix_trials` paths.

```bash
# 1. Make the synthetic data. The output is local parquet, and git ignores it.
uv run python data/generate.py --domain finance
uv run python data/validate.py --domain finance

# 2. Upload to S3 and register the tables in the AWS Glue Data Catalog.
uv run python load/load_athena.py   --domain finance
uv run python load/verify_athena.py --domain finance

# 3. If you want experiment tracking, start the MLflow tracking server.
uv run python ml/mlflow_server.py start

# 4. Train the model and host it. Each feature set becomes one MLflow run.
uv run python ml/train_and_deploy.py --domain finance
uv run python ml/smoke_test.py       --domain finance

# 5. Open analysis/walkthrough_finance.qmd in Positron. Run it cell by cell.

# 6. Render the report and publish it.
uv run quarto render reports/aurora_lending/portfolio_risk_review.qmd
bash setup/publish.sh finance

# 7. Remove the endpoint and stop the MLflow server. Both stop billing.
uv run python ml/teardown.py --all
```

One command checks all of the above:

```bash
bash setup/verify-env.sh finance
```

CAUTION: Two resources bill by the hour. An endpoint bills for every hour that it
exists, even when nothing calls it. An MLflow server bills $0.60 for every hour
that it runs. Run `ml/teardown.py --all` when you finish. That command removes
the endpoints and stops the tracking server. The server is stopped and not
removed, so every run stays. A stopped server bills only for storage, at $0.10
for each GB in a month.

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
ml/teardown.py                  removes endpoints and stops the MLflow server
reports/                        the Quarto reports for Posit Connect
reports/requirements.txt        the packages needed to render, and no more
tests/test_report_config.py     makes sure the reports agree with config.py
iam/                            IAM policy templates and a script to apply them
setup/bootstrap_aws.py          prepares a fresh AWS account
setup/verify-env.sh             checks the environment in one command
setup/publish.sh                renders and publishes to Connect
```

The walkthroughs are Quarto documents, not notebooks, because the SageMaker image
removes every Jupyter kernelspec. Positron runs a Quarto code cell straight into
the console, so a walkthrough still fills the Variables pane, the Plots pane, and
the Data Explorer. You can also render the whole file. If the endpoint is absent,
the model sections print what to deploy and the rest still runs.

## The data

`data/generate.py` makes all of the data from a fixed seed, so every run gives
the same result. The files go to `data/synthetic-<database>/`, and git ignores
them. If the data is absent, each script tells you which command to run.

The row counts are small for a reason. There are 50,000 loans and 1,200 trial
subjects. Athena answers in about one second, so you are never waiting.

`data/data-dict.yaml` describes every column. The terms below are the ones a
reader outside the industry will not know.

### Aurora Lending Group

An unsecured consumer lender. The book holds 50,000 loans from 2019 to 2025,
across 20,000 borrowers and 8 products.

- **Charge-off** means a loan written off as a loss, normally after 120 to 180
  days of non-payment. The rate here is 9.8%, which is realistic.
- **FICO** is the US credit score, from 300 to 850. Aurora refuses applicants
  below 580, so the book holds nothing below that.
- **APR** comes from risk-based pricing. A weaker score pays more.
- **DTI** is debt-to-income, the part of monthly income that pays debt.
- **Decile lift** is how a risk team judges a model. Sort by predicted risk, cut
  into ten equal groups, then compare the prediction against the result.

### Helix Therapeutics

Study HLX-301, a randomized placebo-controlled Phase III trial. It has 1,200
subjects, 45 sites, 4 regions, and a 12-visit schedule from Screening to Week 60.

- **DAS** is the disease activity score, from 0 to 10, and a **lower** score is
  better. Treatment separates from placebo by 1.48 points at Week 24.
- **Arm** is the assignment, Treatment or Placebo, at a ratio of 1 to 1.
- **ECOG performance status**, 0 to 2 here, measures how well a patient does
  daily activities. A higher number is worse.
- **Early discontinuation** means a subject leaves before the end. The rate is
  16.2%. Every subject who leaves costs statistical power.
- An **adverse event** is any unwanted medical event during the study. It does
  not prove the drug caused it.

The prediction target sits on `dim_subject`, not on the fact table. Group the
visit history over a fixed early window, `visit_id <= 5`, before you join. This
avoids two errors that both change the answer:

- A join on a later visit removes every subject who already left. That is
  survivorship bias, and it reports dropout near 6.5% instead of 16.2%.
- A count of missed visits across the whole study depends on exposure. A subject
  who leaves early has fewer visits to miss, so the coefficient gets the wrong
  sign.

## Why it is built this way

- **There is no SageMaker training job.** These datasets fit a logistic
  regression in under a second. A job adds 4 to 8 minutes and changes nothing.
  SageMaker still hosts the model, which is the point.
- **The model is a JSON scorecard, not a pickle.** `ml/entrypoint/inference.py`
  uses only numpy and reads coefficients from JSON. This removes every
  scikit-learn version link between the machine that trains and the container
  that serves. That link is the usual cause of a model that works locally and
  returns HTTP 500 on an endpoint.
- **Logistic regression, not gradient boosting.** For a credit scorecard and for
  clinical risk it is the standard, and a person can read it. The table of
  coefficients is useful content by itself.
- **The `sagemaker` Python SDK is not a dependency.** Version 3 installs torch
  and mlflow, about 2 GB, for features this project does not use.
  `boto3.client("sagemaker-runtime")` is all that inference needs.
- **The MLflow runs compare feature sets, not tuning knobs.** A grid over the
  regularization strength returned the same AUC to four decimal places every
  time. Feature sets answer a question the business asks.
- **Tracking never blocks you.** A `boto3` status check runs first and answers in
  about a second. The MLflow client alone needs more than four minutes to report
  that a server is absent.
- **The reports have their own `requirements.txt`.** Connect rebuilds the
  environment from the bundle, so the render-time list is fixed here instead of
  read from `pyproject.toml`.
- **A report survives the removal of an endpoint.** The model section becomes a
  short note. Every Athena section still renders.
- **pandas, not polars**, because `awswrangler` and `scikit-learn` both use it.

## Configuration

This repository holds no AWS account number. `config.py` reads the account from
AWS STS and builds the bucket name from it. The reports do the same. The IAM
files are templates, with `ACCOUNT_ID` and `REGION` placeholders. A fork works
without an edit.

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

**`iam/sagemaker-demo-athena-access.template.json`** goes on the SageMaker
execution role. `setup/bootstrap_aws.py` applies it. To apply it by hand:

```bash
bash iam/apply-policy.sh sagemaker-demo-athena-access sagemaker-demo-execution-role
```

`AmazonSageMakerFullAccess` already grants S3 access, because the bucket name
contains `sagemaker`, and it grants `sagemaker:InvokeEndpoint`. It grants no
Athena action, only `glue:GetTable*`, and no `sagemaker-mlflow` action at all.
This template adds the Athena reads, the Glue reads, and the MLflow permissions.

**`iam/connect-reader-policy.template.json`** is for the principal that Posit
Connect uses. It is read-only, and it covers the two databases, the bucket, and
`sagemaker:InvokeEndpoint` on the two endpoints.

Two details are easy to get wrong:

- `athena:ListWorkGroups`, `athena:ListDataCatalogs` and
  `athena:ListEngineVersions` are account-level actions. They do not accept a
  resource ARN. If you scope them to a workgroup ARN, AWS denies them.
- **The policy is read-only, so queries must not use CTAS.**
  `awswrangler.athena.read_sql_query` uses `ctas_approach=True` by default, which
  makes a temporary Glue table for every query. That needs `glue:CreateTable`.
  Every query here passes `ctas_approach=False`. If you add a query and forget,
  it works for a user with broad permissions and fails for the demo role.

## Experiment tracking

Amazon SageMaker hosts a managed MLflow tracking server. Training logs one run
for each feature set, so the UI shows a comparison and not a single row.

```bash
uv run python ml/mlflow_server.py create   # once, about 22 minutes
uv run python ml/mlflow_server.py start    # when you want tracking
uv run python ml/mlflow_server.py url      # open the UI
uv run python ml/mlflow_server.py stop     # when you are done
```

Stopping took more than 10 minutes, so start the stop and do not wait for it.
Tracking is optional. If the server is stopped or absent, training prints the
reason and continues.

## Publish to Connect

Each report is a self-contained document. The `.qmd` file declares the settings
it needs instead of importing `config.py`, because the Connect bundle does not
hold `config.py`. `tests/test_report_config.py` fails if the two disagree, and
`setup/publish.sh` runs that test before every deploy.

Connect renders the report against live Athena, so it needs AWS credentials as
content variables. Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_DEFAULT_REGION` for a principal that holds the Connect reader policy. The
`POSIT_DEMO_*` variables also work there.

If Connect has no AWS access, publish the rendered file instead:

```bash
uv run quarto publish connect reports/aurora_lending/portfolio_risk_review.qmd
```

The report then cannot refresh itself, and nothing else changes.

`_brand.yml` holds the color palette that both reports use. Point it at another
brand file to change the appearance.

## When something is wrong

| What you see | Cause | What to do |
|---|---|---|
| `AccessDeniedException` from an Athena call | the Athena policy is absent | Run `setup/bootstrap_aws.py`, or `iam/apply-policy.sh`. |
| `AccessDeniedException` naming `glue:CreateTable` | a query uses the default CTAS path | Pass `ctas_approach=False`. The role is read-only by design. |
| `AccessDeniedException` from an MLflow call | the `sagemaker-mlflow` permissions are absent | Run `setup/bootstrap_aws.py`. `AmazonSageMakerFullAccess` grants none of them. |
| Every Athena query reports a missing output location | staging was not passed | Workgroup `primary` has no default. Use `config.athena_staging()`. |
| `quarto render` reports that Jupyter is absent | Quarto used the system Python | Run `uv run quarto render`, not `quarto render`. |
| The endpoint returns HTTP 5xx | the serving entrypoint is broken | `ml/smoke_test.py` reports this. Make sure the archive holds `scorecard.json` at the root and `code/inference.py`. |
| The walkthrough says the MLflow server is not running | the server is stopped | Run `uv run python ml/mlflow_server.py start`. |
| The report renders, but the model section is a note | the endpoint was removed | This is correct behavior. Deploy it again if you need that section. |
| Positron never opens, but the app looks healthy | the license check failed | Licensing uses AWS License Manager. A session that loses its license stops after 10 minutes. |

## What is not yet tested

- **The endpoint path has never run from start to finish.** Model training, the
  archive layout, and the scoring arithmetic are tested. The container that loads
  `code/inference.py` is not. Treat your first `train_and_deploy.py` as a
  rehearsal.
- **No report has gone to Connect.** The bundle renders on its own with no access
  to the repository, which is the difficult part, but no deploy has run.

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
