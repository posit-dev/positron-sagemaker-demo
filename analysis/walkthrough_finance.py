# %% [markdown]
# # Aurora Lending Group: portfolio risk walkthrough
#
# **This project contains synthetic data and analysis created for demonstration
# purposes only.**
#
# Run this file one cell at a time in Positron on Amazon SageMaker. Use
# Ctrl+Enter, or Cmd+Enter on a Mac. Each `# %%` block is one step:
#
# 1. we are already authenticated
# 2. the governed catalog
# 3. a real Athena query
# 4. the Data Explorer
# 5. a chart
# 6. a model hosted by SageMaker
# 7. the result

# %%
# --- 1. No keys and no setup ----------------------------------------------
# The Positron image holds an AWS profile that asks the Studio app for
# credentials. boto3 therefore finds the SageMaker execution role by itself.
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import awswrangler as wr
import boto3
import matplotlib.pyplot as plt
import pandas as pd

import config

session = boto3.Session(region_name=config.REGION)
identity = session.client("sts").get_caller_identity()
print(f"account {identity['Account']}")
print(f"role    {identity['Arn'].split('/')[-2] if '/' in identity['Arn'] else identity['Arn']}")
print(f"region  {config.REGION}")

# %%
# --- 2. The governed catalog ----------------------------------------------
# AWS Glue is the catalog. Athena is the engine. Nothing was copied here.
wr.catalog.tables(database="aurora_lending", boto3_session=session)[
    ["Table", "Description", "TableType"]
]

# %%
# --- 3. A query, straight into a DataFrame --------------------------------
# The workgroup "primary" has no default output location. Every query must
# therefore pass a staging location, or Athena returns an error.
loans = wr.athena.read_sql_query(
    """
    SELECT f.loan_id,
           f.origination_date,
           f.fico_at_origination,
           f.dti,
           f.apr,
           f.term_months,
           CAST(f.principal_amount AS double)    AS principal_amount,
           CAST(f.outstanding_balance AS double) AS outstanding_balance,
           f.employment_length_years,
           f.purpose,
           f.charged_off,
           b.state,
           b.income_band,
           b.homeownership,
           p.product_name
    FROM fct_loan_performance f
    JOIN dim_borrower b     ON b.borrower_id = f.borrower_id
    JOIN dim_loan_product p ON p.product_id  = f.product_id
    """,
    database="aurora_lending",
    workgroup=config.ATHENA_WORKGROUP,
    s3_output=config.athena_staging(),
    # The default option makes a temporary Glue table. The demo role is
    # read-only and cannot do that.
    ctas_approach=False,
    boto3_session=session,
)
print(f"{len(loans):,} loans and {len(loans.columns)} columns")
loans.head()

# %%
# --- 4. Open this in the Data Explorer ------------------------------------
# Click `loans` in the Variables pane. The Data Explorer then sorts any column,
# filters rows, and shows a summary of every column, with no code. Take your
# time on this step.
loans.describe()

# %%
# --- 5. Where is the risk? -----------------------------------------------
# Charge-off rate by FICO band, and by year of origination.
loans["fico_band"] = pd.cut(
    loans["fico_at_origination"],
    bins=[579, 620, 660, 700, 740, 780, 850],
    labels=["580-619", "620-659", "660-699", "700-739", "740-779", "780+"],
)
loans["vintage"] = pd.to_datetime(loans["origination_date"]).dt.year

by_band = loans.groupby("fico_band", observed=True)["charged_off"].mean() * 100
by_vintage = (
    loans.pivot_table(index="vintage", columns="fico_band",
                      values="charged_off", observed=True) * 100
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
by_band.plot.bar(ax=ax1, color="#447099", rot=0)
ax1.set(title="Charge-off rate by FICO band", xlabel="FICO at origination",
        ylabel="charge-off rate (%)")
by_vintage.plot(ax=ax2, marker="o", linewidth=1.6, colormap="viridis")
ax2.set(title="Charge-off rate by year", xlabel="year of origination",
        ylabel="charge-off rate (%)")
ax2.legend(title="FICO band", fontsize=7, ncol=2)
fig.tight_layout()
plt.show()

# %%
# --- 6. Score the book with the hosted model -----------------------------
# SageMaker hosts the model on a real-time endpoint. This cell sends an HTTPS
# request, which the same execution role authorizes. Nothing loads locally.
import json

FEATURES = ["fico_at_origination", "dti", "apr", "term_months",
            "principal_amount", "employment_length_years", "purpose"]

sample = loans.sample(1_000, random_state=20260821)
runtime = session.client("sagemaker-runtime")

try:
    response = runtime.invoke_endpoint(
        EndpointName=config.FINANCE.endpoint,
        ContentType="application/json",
        Body=json.dumps({"instances": sample[FEATURES].to_dict(orient="records")}),
    )
    sample["risk_score"] = json.loads(response["Body"].read())["predictions"]
    print(f"scored {len(sample):,} loans on {config.FINANCE.endpoint}")
except runtime.exceptions.ClientError as err:
    raise SystemExit(
        f"The endpoint {config.FINANCE.endpoint} did not answer: {err}\n"
        "Deploy it first:  uv run python ml/train_and_deploy.py --domain finance"
    ) from None

sample[["loan_id", "fico_at_origination", "dti", "purpose", "risk_score", "charged_off"]].head(10)

# %%
# --- 7. Does the model rank the risk? ------------------------------------
# Sort by predicted risk, cut into ten equal groups, then compare the
# prediction against the charge-off rate that each group reached.
sample["decile"] = pd.qcut(sample["risk_score"], 10, labels=False, duplicates="drop") + 1
lift = sample.groupby("decile").agg(
    loans=("loan_id", "size"),
    mean_predicted=("risk_score", "mean"),
    actual_chargeoff=("charged_off", "mean"),
    exposure=("outstanding_balance", "sum"),
)
lift[["mean_predicted", "actual_chargeoff"]] *= 100
print(lift.round(2).to_string())

# Compare against the base rate, not against the safest decile. The safest
# decile can hold zero charge-offs, and that gives a meaningless ratio.
base_rate = sample["charged_off"].mean() * 100
top_rate = lift.loc[10, "actual_chargeoff"]
captured = (
    sample.loc[sample["decile"] >= 9, "charged_off"].sum()
    / max(sample["charged_off"].sum(), 1) * 100
)
print(f"\nbase rate              {base_rate:.1f}%")
print(f"riskiest decile        {top_rate:.1f}%  ({top_rate / base_rate:.1f}x base rate)")
print(f"top 2 deciles hold     {captured:.0f}% of all charge-offs")

# %%
# --- What comes next -----------------------------------------------------
# The same query and the same endpoint drive the published report:
#
#   uv run quarto render reports/aurora_lending/portfolio_risk_review.qmd
#   bash setup/publish.sh finance
#
# After the session, stop the endpoint billing:
#
#   uv run python ml/teardown.py --all
