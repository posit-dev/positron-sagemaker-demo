# %% [markdown]
# # Helix Therapeutics -- study HLX-301 enrolment and retention walkthrough
#
# **This project contains synthetic data and analysis created for demonstration
# purposes only.**
#
# Run this file cell by cell in Positron on Amazon SageMaker (Ctrl/Cmd+Enter).
# Study HLX-301 is a randomised, placebo-controlled Phase III trial; the
# efficacy measure is a disease activity score (DAS), where lower is better.
#
# The operational question: **as of Week 12, which enrolled subjects are at risk
# of discontinuing, so sites can intervene?**

# %%
# --- 1. No keys, no setup -------------------------------------------------
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
# --- 2. The governed catalogue --------------------------------------------
wr.catalog.tables(database="helix_trials", boto3_session=session)[["Table", "TableType"]]

# %%
# --- 3. Subject-level query ----------------------------------------------
# Visit history is aggregated over a FIXED early window (Screening..Week 12)
# before joining. Two reasons, both of which matter clinically:
#   * joining on a single later visit would silently exclude everyone who left
#     before it -- survivorship bias, and it understates dropout badly
#   * a raw count over the whole study is confounded by exposure, since subjects
#     who leave early have fewer visits available to miss
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
    boto3_session=session,
    # ctas_approach=False: the default creates temporary Glue tables, which
    # needs Glue write permission the read-only demo role does not have.
    ctas_approach=False,
)
print(f"{len(subjects):,} subjects x {len(subjects.columns)} columns")
print(f"discontinuation rate  {subjects['discontinued'].mean() * 100:.1f}%")
subjects.head()

# %%
# --- 4. Open this in the Data Explorer ------------------------------------
# Click `subjects` in the Variables pane. Sort by missed_visit_rate, filter to
# arm == "Treatment", and read the column summaries -- no code required.
subjects.groupby("arm")[["age", "baseline_das", "early_ae_count", "missed_visit_rate"]].mean().round(2)

# %%
# --- 5. Efficacy and safety at a glance ----------------------------------
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
    boto3_session=session,
    # ctas_approach=False: the default creates temporary Glue tables, which
    # needs Glue write permission the read-only demo role does not have.
    ctas_approach=False,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for arm, colour in (("Treatment", "#447099"), ("Placebo", "#ee6331")):
    part = das[das["arm"] == arm]
    ax1.plot(part["scheduled_day"], part["mean_das"], marker="o", label=arm, color=colour)
ax1.set(title="Disease activity score by arm (lower is better)",
        xlabel="study day", ylabel="mean DAS")
ax1.legend()

by_ae = subjects.groupby("early_ae_count")["discontinued"].mean() * 100
by_ae.plot.bar(ax=ax2, color="#9a4665", rot=0)
ax2.set(title="Discontinuation by adverse events through Week 12",
        xlabel="adverse events recorded by Week 12", ylabel="discontinued (%)")
fig.tight_layout()
plt.show()

# %%
# --- 6. Score the active cohort against the hosted model -----------------
# The dropout-risk model is hosted on a SageMaker real-time endpoint.
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
        f"Endpoint {config.LIFESCI.endpoint} is not available ({err}).\n"
        "Deploy it first:  uv run python ml/train_and_deploy.py --domain lifesci"
    ) from None

# %%
# --- 7. Which sites need a retention conversation? -----------------------
# This is the deliverable a study manager would actually act on.
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
print("highest predicted-dropout sites (>= 20 subjects enrolled):")
print(watchlist.head(10).round(1).to_string())

flagged = subjects.nlargest(25, "dropout_risk")
print(f"\ntop 25 individual subjects by risk: "
      f"{flagged['discontinued'].mean() * 100:.0f}% actually discontinued, "
      f"against a {subjects['discontinued'].mean() * 100:.1f}% study-wide rate")

# %%
# --- Hand-off ------------------------------------------------------------
#   quarto render reports/helix_trials/enrollment_safety_review.qmd
#   bash setup/publish.sh --domain lifesci
#   uv run python ml/teardown.py --all
