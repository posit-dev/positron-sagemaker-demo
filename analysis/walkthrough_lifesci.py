# %% [markdown]
# # Helix Therapeutics: study HLX-301 enrollment and retention walkthrough
#
# **This project contains synthetic data and analysis created for demonstration
# purposes only.**
#
# Run this file one cell at a time in Positron on Amazon SageMaker. Use
# Ctrl+Enter, or Cmd+Enter on a Mac.
#
# Study HLX-301 is a randomized, placebo-controlled Phase III trial. The
# efficacy measure is a disease activity score, or DAS, where a lower score is
# better.
#
# The question a study team asks: **at Week 12, which subjects are likely to
# leave the study, so that sites can act?**

# %%
# --- 1. No keys and no setup ----------------------------------------------
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
print(f"region  {config.REGION}")

# %%
# --- 2. The governed catalog ----------------------------------------------
wr.catalog.tables(database="helix_trials", boto3_session=session)[["Table", "TableType"]]

# %%
# --- 3. One row for each subject -----------------------------------------
# Group the visit history over a fixed early window first, Screening to Week 12,
# and then join. Two errors are avoided this way, and both matter clinically.
#
# If you join on a later visit first, the query removes every subject who
# already left. That is survivorship bias, and it reports dropout near 7%
# instead of 16%.
#
# A raw count over the whole study depends on exposure. A subject who leaves
# early has fewer visits available to miss, so the coefficient gets the wrong
# sign.
subjects = wr.athena.read_sql_query(
    """
    WITH early_window AS (
        SELECT subject_id,
               count(*)                                                  AS early_visits,
               sum(CASE WHEN adverse_event_flag THEN 1 ELSE 0 END)        AS early_ae,
               CAST(sum(CASE WHEN NOT visit_completed THEN 1 ELSE 0 END) AS double)
                   / count(*)                                            AS missed_visit_rate
        FROM fct_visit_observations
        WHERE visit_id <= 5
        GROUP BY subject_id
    )
    SELECT s.subject_id, s.arm, s.age, s.sex, s.ecog_status, s.bmi,
           s.prior_therapy_lines, s.baseline_das, s.early_ae_count,
           w.missed_visit_rate, s.discontinued,
           t.site_name, t.country, t.region
    FROM dim_subject s
    JOIN early_window w ON w.subject_id = s.subject_id
    JOIN dim_site t     ON t.site_id    = s.site_id
    """,
    database="helix_trials",
    workgroup=config.ATHENA_WORKGROUP,
    s3_output=config.athena_staging(),
    # The default option makes a temporary Glue table. The demo role is
    # read-only and cannot do that.
    ctas_approach=False,
    boto3_session=session,
)
print(f"{len(subjects):,} subjects and {len(subjects.columns)} columns")
print(f"discontinuation rate  {subjects['discontinued'].mean() * 100:.1f}%")
subjects.head()

# %%
# --- 4. Open this in the Data Explorer ------------------------------------
# Click `subjects` in the Variables pane. Sort by missed_visit_rate, filter the
# arm to Treatment, and read the column summaries. No code is necessary.
subjects.groupby("arm")[["age", "baseline_das", "early_ae_count", "missed_visit_rate"]].mean().round(2)

# %%
# --- 5. Efficacy and safety ----------------------------------------------
das = wr.athena.read_sql_query(
    """
    SELECT v.visit_name, v.scheduled_day, s.arm,
           avg(o.das_score)      AS mean_das,
           count(*)              AS observations
    FROM fct_visit_observations o
    JOIN dim_subject s ON s.subject_id = o.subject_id
    JOIN dim_visit   v ON v.visit_id   = o.visit_id
    WHERE v.scheduled_day >= 0
    GROUP BY v.visit_name, v.scheduled_day, s.arm
    ORDER BY v.scheduled_day
    """,
    database="helix_trials",
    workgroup=config.ATHENA_WORKGROUP,
    s3_output=config.athena_staging(),
    ctas_approach=False,
    boto3_session=session,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for arm, color in (("Treatment", "#447099"), ("Placebo", "#ee6331")):
    part = das[das["arm"] == arm]
    ax1.plot(part["scheduled_day"], part["mean_das"], marker="o", label=arm, color=color)
ax1.set(title="Disease activity score by arm, lower is better",
        xlabel="study day", ylabel="mean DAS")
ax1.legend()

by_ae = subjects.groupby("early_ae_count")["discontinued"].mean() * 100
by_ae.plot.bar(ax=ax2, color="#9a4665", rot=0)
ax2.set(title="Discontinuation by adverse events to Week 12",
        xlabel="adverse events recorded by Week 12", ylabel="discontinued (%)")
fig.tight_layout()
plt.show()

# %%
# --- 6. Score the cohort with the hosted model ---------------------------
# SageMaker hosts the dropout-risk model on a real-time endpoint.
import json

FEATURES = ["age", "ecog_status", "bmi", "prior_therapy_lines",
            "baseline_das", "early_ae_count", "missed_visit_rate", "arm"]

runtime = session.client("sagemaker-runtime")
try:
    response = runtime.invoke_endpoint(
        EndpointName=config.LIFESCI.endpoint,
        ContentType="application/json",
        Body=json.dumps({"instances": subjects[FEATURES].to_dict(orient="records")}),
    )
    subjects["dropout_risk"] = json.loads(response["Body"].read())["predictions"]
    print(f"scored {len(subjects):,} subjects on {config.LIFESCI.endpoint}")
except runtime.exceptions.ClientError as err:
    raise SystemExit(
        f"The endpoint {config.LIFESCI.endpoint} did not answer: {err}\n"
        "Deploy it first:  uv run python ml/train_and_deploy.py --domain lifesci"
    ) from None

# %%
# --- 7. Which sites need a retention conversation? -----------------------
# This table is what a study manager acts on.
watchlist = (
    subjects.groupby(["region", "site_name"])
    .agg(subjects_enrolled=("subject_id", "size"),
         mean_dropout_risk=("dropout_risk", "mean"),
         actual_discontinued=("discontinued", "mean"))
    .assign(mean_dropout_risk=lambda d: d["mean_dropout_risk"] * 100,
            actual_discontinued=lambda d: d["actual_discontinued"] * 100)
    .query("subjects_enrolled >= 20")
    .sort_values("mean_dropout_risk", ascending=False)
)
print("sites with the highest predicted dropout, 20 or more subjects enrolled:")
print(watchlist.head(10).round(1).to_string())

flagged = subjects.nlargest(25, "dropout_risk")
print(f"\nthe 25 highest-risk subjects: "
      f"{flagged['discontinued'].mean() * 100:.0f}% left the study, "
      f"against {subjects['discontinued'].mean() * 100:.1f}% for the whole study")

# %%
# --- What comes next -----------------------------------------------------
#   uv run quarto render reports/helix_trials/enrollment_safety_review.qmd
#   bash setup/publish.sh lifesci
#   uv run python ml/teardown.py --all
