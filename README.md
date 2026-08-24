# Positron on Amazon SageMaker

A data science workflow that runs inside Positron on Amazon SageMaker. It reads
governed data from Amazon Athena, scores that data with a model hosted on a
SageMaker endpoint, and publishes a report to Posit Connect. All of it happens in
one IDE.

This repository supports joint demonstrations by Posit and AWS. All data in it is
synthetic.

If you present this demo, read [PRESENTING.md](PRESENTING.md) first. It gives the
industry background, the steps to do before a session, what to say at each step,
and the numbers you can quote.

## The two demos

The repository holds two demos. They are alternatives for different sessions, not
two halves of one story. Every script takes an explicit `--domain` so that
nothing runs both by accident.

| Domain | Company | Question | Athena database |
|---|---|---|---|
| `finance` | Aurora Lending Group | Which consumer loans will charge off? | `aurora_lending` |
| `lifesci` | Helix Therapeutics | Which trial subjects will leave the study? | `helix_trials` |

Both demos predict a yes-or-no outcome. This is deliberate. The modeling code and
the endpoint code are the same in both, so the audience learns one pattern.

## What you need

- Positron on Amazon SageMaker. This is a JupyterLab app that runs the `positron-sagemaker` custom image.
- `uv` and `quarto`. The image contains both.
- An execution role that can read the demo databases through Athena. Read [IAM](#iam) below.
- A Posit Connect server, if you want to publish the report.

## Install

```bash
uv sync
```

The SageMaker image contains few Python packages. This project brings its own
environment instead of depending on the image.

## Run a demo

Select one domain. Then do the steps in order. The example uses `finance`. For
the other demo, use `--domain lifesci` and the `helix_trials` paths.

```bash
# 1. Make the synthetic data. The output is local parquet, and git ignores it.
uv run python data/generate.py --domain finance
uv run python data/validate.py --domain finance

# 2. Upload to S3 and register the tables in the AWS Glue Data Catalog.
uv run python load/load_athena.py   --domain finance
uv run python load/verify_athena.py --domain finance

# 3. Train the model and host it on a SageMaker endpoint.
uv run python ml/train_and_deploy.py --domain finance
uv run python ml/smoke_test.py       --domain finance

# 4. Open analysis/walkthrough_finance.qmd in Positron. Run it cell by cell.

# 5. Render the report and publish it.
uv run quarto render reports/aurora_lending/portfolio_risk_review.qmd
bash setup/publish.sh finance

# 6. Remove the endpoint so that it stops billing.
uv run python ml/teardown.py --all
```

One command checks all of the above:

```bash
bash setup/verify-env.sh finance
```

CAUTION: A real-time endpoint bills for every hour that it exists, even when
nothing calls it. Always run `ml/teardown.py --all` after a session.

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
ml/train_and_deploy.py          trains the model and hosts it on SageMaker
ml/entrypoint/inference.py      the serving entrypoint, which uses only numpy
ml/smoke_test.py                makes sure a live endpoint scores correctly
ml/teardown.py                  removes endpoints so that they stop billing
reports/                        the Quarto reports for Posit Connect
reports/requirements.txt        the packages needed to render, and no more
tests/test_report_config.py     makes sure the reports agree with config.py
PRESENTING.md                   the guide for whoever presents the demo
iam/                            IAM policy templates and a script to apply them
setup/verify-env.sh             one check before a session
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
subjects. Athena answers in about one second, so nothing feels slow in front of
an audience.

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

The pieces work on their own. Three are useful well beyond this demo:

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
