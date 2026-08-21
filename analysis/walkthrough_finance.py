# %% [markdown]
# # Aurora Lending Group -- portfolio risk walkthrough
#
# **This project contains synthetic data and analysis created for demonstration
# purposes only.**
#
# Run this file cell by cell in Positron on Amazon SageMaker (Ctrl/Cmd+Enter).
# Each `# %%` block is one beat of the story:
#
# 1. we are already authenticated
# 2. the governed catalogue
# 3. a real Athena query
# 4. the Data Explorer
# 5. a chart
# 6. a SageMaker-hosted model
# 7. hand off to the report

# %%
# --- 1. No keys, no setup -------------------------------------------------
# The Positron image ships an AWS profile whose credential_process shells out to
# the Studio app, so boto3 resolves the SageMaker execution role automatically.
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
# --- 2. The governed catalogue --------------------------------------------
# Glue is the catalogue; Athena is the engine. Nothing was copied or extracted.
wr.catalog.tables(database="aurora_lending", boto3_session=session)[
    ["Table", "Description", "TableType"]
]

# %%
# --- 3. A real query, straight into a DataFrame ---------------------------
# Workgroup "primary" has no default output location, so staging is passed
# explicitly -- otherwise every query errors.
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
    boto3_session=session,
    # ctas_approach=False: the default creates temporary Glue tables, which
    # needs Glue write permission the read-only demo role does not have.
    ctas_approach=False,
)
print(f"{len(loans):,} loans x {len(loans.columns)} columns")
loans.head()

# %%
# --- 4. Open this in the Data Explorer ------------------------------------
# In the Variables pane, click `loans` to open Positron's Data Explorer: sort by
# any column, filter, and read the per-column distribution summaries without
# writing a line of code. This is the beat worth slowing down for.
loans.describe()

# %%
# --- 5. Where is the risk concentrated? ----------------------------------
# Charge-off rate by FICO band and origination vintage.
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
ax2.set(title="Charge-off rate by vintage", xlabel="origination year",
        ylabel="charge-off rate (%)")
ax2.legend(title="FICO band", fontsize=7, ncol=2)
fig.tight_layout()
plt.show()

# %%
# --- 6. Score the book against the hosted model --------------------------
# The model lives on a SageMaker real-time endpoint. Nothing is loaded locally;
# this is an HTTPS call authorised by the same execution role.
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
        f"Endpoint {config.FINANCE.endpoint} is not available ({err}).\n"
        "Deploy it first:  uv run python ml/train_and_deploy.py --domain finance"
    ) from None

sample[["loan_id", "fico_at_origination", "dti", "purpose", "risk_score", "charged_off"]].head(10)

# %%
# --- 7. Does the model actually rank risk? -------------------------------
# Decile lift: sort by predicted risk, then check the realised charge-off rate.
sample["decile"] = pd.qcut(sample["risk_score"], 10, labels=False, duplicates="drop") + 1
lift = sample.groupby("decile").agg(
    loans=("loan_id", "size"),
    mean_predicted=("risk_score", "mean"),
    actual_chargeoff=("charged_off", "mean"),
    exposure=("outstanding_balance", "sum"),
)
lift[["mean_predicted", "actual_chargeoff"]] *= 100
print(lift.round(2).to_string())

# Stated against the base rate rather than the safest decile: the safest decile
# can legitimately contain zero charge-offs, and dividing by that produces a
# meaningless ratio.
base_rate = sample["charged_off"].mean() * 100
top_rate = lift.loc[10, "actual_chargeoff"]
captured = (
    sample.loc[sample["decile"] >= 9, "charged_off"].sum()
    / max(sample["charged_off"].sum(), 1) * 100
)
print(f"\nbase rate              {base_rate:.1f}%")
print(f"riskiest decile        {top_rate:.1f}%  ({top_rate / base_rate:.1f}x base rate)")
print(f"top 2 deciles capture  {captured:.0f}% of all charge-offs")

# %%
# --- Hand-off ------------------------------------------------------------
# The same Athena query and the same endpoint drive the published report:
#
#   quarto render reports/aurora_lending/portfolio_risk_review.qmd
#   bash setup/publish.sh --domain finance
#
# When the session is over, stop the endpoint billing:
#
#   uv run python ml/teardown.py --all
