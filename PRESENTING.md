# Presenter guide

Ten minutes with this document is enough to run either demo for a customer. Read
the section for the demo you present. Skip the other one.

There are two demos, for two different sessions. Do not present both. Each one
works from start to finish on its own.

## 1. What the audience must take away

Positron is a full IDE inside Amazon SageMaker. The walkthrough proves this in
five steps:

1. **No credential setup.** `boto3` finds the SageMaker execution role by itself.
   The image contains an AWS profile that asks the Studio app for credentials.
   There are no keys and no `aws configure`.
2. **Governed data, not an extract.** AWS Glue is the catalog and Athena is the
   engine. Nothing is copied to the workstation.
3. **The Data Explorer.** This is the most Positron-specific step. Click a
   DataFrame in the Variables pane. You then get sorting, filtering, and a
   summary of every column, with no code. Go slowly here.
4. **A SageMaker service is only an API call.** The model runs on a real-time
   endpoint. The walkthrough calls it over HTTPS with the same role.
5. **Publishing is one command.** The same query and the same endpoint make a
   Quarto report, which goes to Posit Connect.

## 2. AWS resources

| Resource | Value |
|---|---|
| SageMaker domain | `sagemaker-demo`, network mode `PublicInternetOnly` |
| Execution role | `sagemaker-demo-execution-role` |
| Image | the `positron-sagemaker` custom image, run as a JupyterLab app |
| S3 bucket | `sagemaker-posit-conf-2026-demo-<account>-<region>` |
| Glue databases | `aurora_lending` and `helix_trials` |
| Athena workgroup | `primary` |
| Endpoints | `aurora-lending-risk` and `helix-dropout-risk`, made when needed |

The bucket name starts with `sagemaker` for a reason. `AmazonSageMakerFullAccess`
grants S3 object access on `arn:aws:s3:::*sagemaker*`, so no bucket-specific S3
policy is necessary.

The Athena workgroup `primary` has no default output location. Every client must
pass staging explicitly, and `config.athena_staging()` does this.

Lake Formation uses `IAM_ALLOWED_PRINCIPALS`, so plain IAM controls Glue. No Lake
Formation grant is necessary.

The one policy that had to be added is `PositConf2026AthenaAccess`. It grants
Athena queries and Glue reads, and no Glue write. For this reason every
`read_sql_query` call in the repository passes `ctas_approach=False`. The default
makes a temporary Glue table, and AWS denies that.

## 3. Data context

A presenter is usually not a subject-matter expert in either field. This section
is enough to answer a question with confidence.

### Aurora Lending Group (financial services)

An unsecured consumer lender. The products are personal loans, credit card
refinancing, home improvement, auto refinance, medical, small business, and
education. The book holds 50,000 loans from 2019 to 2025, across 20,000 borrowers
and 8 products.

- **Charge-off** is the industry term for a loan written off as a loss. This
  normally happens after 120 to 180 days of non-payment. The rate here is 9.8%,
  which is realistic for unsecured consumer credit.
- **FICO** is the US credit score, from 300 to 850. Aurora refuses applicants
  below 580, so the book contains nothing below that. Mention this if somebody
  asks why there is no deep subprime tail.
- **APR** comes from risk-based pricing. A weaker credit score pays more.
- **DTI** is debt-to-income. It is the part of monthly income that pays debt.
- **Decile lift** is how a risk team judges a model. Sort by predicted risk, cut
  into ten equal groups, then compare the prediction against the result.

### Helix Therapeutics (life sciences)

Study HLX-301, a randomized placebo-controlled Phase III trial. It has 1,200
subjects, 45 sites, 4 regions, and a 12-visit schedule from Screening to Week 60.

- **DAS** is the disease activity score, from 0 to 10. It measures how well the
  drug works, and a **lower** score is better. The treatment arm separates from
  placebo by 1.48 points at Week 24.
- **Arm** is the assignment, either Treatment or Placebo, at a ratio of 1 to 1.
- **ECOG performance status**, 0 to 2 here, measures how well a patient can do
  daily activities. A higher number is worse.
- **Early discontinuation** means a subject leaves the study before the end. The
  rate is 16.2%. It matters commercially, because every subject who leaves costs
  statistical power and can put the result of the trial at risk.
- An **adverse event** is any unwanted medical event during the study. It does
  not prove that the drug caused the event.

**One detail is worth knowing.** The prediction target is on `dim_subject`, not
on the fact table. The query must group the visit history over a fixed early
window, `visit_id <= 5`, which is Screening to Week 12, **before** it joins. This
avoids two errors:

- A join on a later visit removes every subject who already left. This is
  survivorship bias, and it reports dropout near 7% instead of 16%.
- A count of missed visits across the whole study depends on exposure. A subject
  who leaves early has fewer visits to miss, and the coefficient then gets the
  wrong sign.

A clinical audience knows both errors. It lands well when you say that the query
is built to avoid them.

## 4. Before a session

Do this about 15 minutes before, inside Positron on SageMaker:

```bash
uv sync
bash setup/verify-env.sh finance
```

Every line must read `[ok]`. If the endpoint line fails, run this:

```bash
uv run python ml/train_and_deploy.py --domain finance
uv run python ml/smoke_test.py       --domain finance
```

The endpoint needs 5 to 8 minutes to become `InService`. Do this before the
session, not during it.

Then open `analysis/walkthrough_finance.qmd` and run cell 1, to start the
Python interpreter. To make sure Connect is registered, run
`uv tool run --from rsconnect-python rsconnect list`.

CAUTION: After the session, always run `uv run python ml/teardown.py --all`. A
real-time endpoint bills for every hour that it exists, even when nothing calls
it.

## 5. The walkthrough

Run `analysis/walkthrough_<domain>.qmd` cell by cell. It takes 10 to 12
minutes. Positron runs a Quarto cell in the console, the same as a script.

| Section | Step | What to say |
|---|---|---|
| 1 | Identity | "I configured no credentials. The image finds the SageMaker execution role for me." |
| 2 | Catalog | "This is Glue, the governed catalog. I am not looking at a copy of the data." |
| 3 | Athena query | "That is a real query against Athena, straight into a DataFrame." |
| 4 | **Data Explorer** | "Let me click into this." Sort, filter, and show the column summaries. **Give this step the most time.** |
| 5 | First chart | "This is ordinary matplotlib, drawn in the Plots pane." |
| 6 | Second chart | Finance shows the balance next to the rate. Life sciences shows retention over time. |
| 7 | Endpoint | "SageMaker hosts the model. I only call it. Nothing loads locally." |
| 8 | Result | Read out the numbers in section 6 of this guide. |

Both walkthroughs have the same eight sections, so the story is the same
whichever demo you present.

Then render and publish:

```bash
uv run quarto render reports/aurora_lending/portfolio_risk_review.qmd
bash setup/publish.sh finance
```

Open the Connect URL. Point out that the report renders itself again against live
Athena on a schedule.

`publish.sh` copies `_brand.yml` and `requirements.txt` next to the `.qmd` file,
then deploys only that directory. The reports are self-contained by design, so
the bundle needs neither `config.py` nor the rest of the repository.

For the refresh to work, Connect needs `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` as content variables, for a
principal that holds `iam/connect-reader-policy.template.json`. If that is not
ready, use `uv run quarto publish connect <qmd>` instead. That command sends the
rendered HTML and needs no AWS access on Connect.

## 6. Numbers you can quote

Make the data again if the seed changed. Do not trust these numbers after a
change.

**Aurora Lending Group**

- 50,000 loans, $1,055M advanced, $423M outstanding
- charge-off rate 9.8%, model test AUC 0.75, which is realistic for credit risk
- the riskiest decile reaches about 32%, near 2.9 times the base rate
- the top two deciles hold about 48% of all charge-offs
- small business is the worst purpose at 17.2%, auto refinance the best at 5.5%

**Helix Therapeutics**

- 1,200 subjects, 45 sites, 16.2% early discontinuation
- model test AUC 0.80
- the 50 highest-risk subjects show 78% discontinuation against 16.2% for the
  study, near 4.8 times enrichment. The top 25 reach 96%.
- DAS at Week 24 is 4.75 for treatment and 6.23 for placebo, a gap of 1.48 points
- dropout rises from 10.3% with no early adverse event to 54.5% with two

## 7. Design decisions you can defend

A technical viewer can ask why the demo works this way.

- **There is no SageMaker training job.** These datasets fit a logistic
  regression in under a second. A training job adds 4 to 8 minutes of preparation
  and changes nothing that the audience sees. SageMaker still *hosts* the model,
  which is the point. `ml/train_and_deploy.py` trains locally, then deploys.
- **The model is a JSON scorecard, not a pickle.** `ml/entrypoint/inference.py`
  uses only numpy and reads coefficients from JSON. This removes every
  scikit-learn version link between the machine that trains and the container
  that serves. That link is the usual cause of a model that works locally and
  returns HTTP 500 on an endpoint.
- **Logistic regression, not gradient boosting.** For a credit scorecard and for
  clinical risk it is the industry standard, and a person can read it. The table
  of coefficients is useful report content by itself.
- **The `sagemaker` Python SDK is not a dependency.** Version 3 installs torch
  and mlflow, about 2 GB, for features this demo does not use.
  `boto3.client("sagemaker-runtime")` is all that inference needs.
- **The reports have their own `requirements.txt`.** Connect rebuilds the
  environment from the bundle. The render-time package list is therefore fixed on
  purpose, instead of read from `pyproject.toml`.
- **A report survives the removal of an endpoint.** If the endpoint is absent,
  the model section becomes a short note. Every Athena section still renders.

### Differences from the standard Posit demo layout

- **pandas, not polars.** `awswrangler` and `scikit-learn` both use pandas.
- **One `_brand.yml` at the top level**, not one for each fictional company. The
  two demos never appear together, so a second palette is invisible complexity.

## 8. When something is wrong

| What you see | Cause | What to do |
|---|---|---|
| `AccessDeniedException` from any Athena call | the `PositConf2026AthenaAccess` policy is absent | Run `bash iam/apply-policy.sh sagemaker-demo-athena-access sagemaker-demo-execution-role`. |
| `AccessDeniedException` that names `glue:CreateTable` | a query uses the default CTAS path | Pass `ctas_approach=False`. The demo role is read-only by design. |
| Every Athena query reports a missing output location | staging was not passed | Workgroup `primary` has no default. Use `config.athena_staging()`. |
| `quarto render` reports that Jupyter is absent | Quarto used the system Python | Run `uv run quarto render`, not `quarto render`. |
| The endpoint returns HTTP 5xx | the serving entrypoint is broken | `ml/smoke_test.py` reports this. Make sure the archive holds `scorecard.json` at the root and `code/inference.py`. |
| Positron never opens, but the app looks healthy | the license check failed | Licensing uses AWS License Manager. A session that loses its license stops after 10 minutes. |
| The report renders, but the model section is a note | the endpoint was removed | This is correct behavior. Deploy the endpoint again if you need that section. |

## 9. What is not yet tested

- **The endpoint path has never run from start to finish.** Model training, the
  archive layout, and the scoring arithmetic are all tested. The container that
  loads `code/inference.py` is not. Treat the first run of
  `train_and_deploy.py` as a rehearsal, and do it well before a session.
- **No report went to Connect yet.** The bundle renders on its own, with no
  access to the repository. That is the difficult part, and it works. But no
  deploy ran, and no Connect credentials were used.
- The image tag in use is a daily preview build, not a fixed release. If
  licensing behaves strangely, look at this first.
