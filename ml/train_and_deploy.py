"""Train a risk model from Athena data, then host it on a SageMaker endpoint.

    uv run python ml/train_and_deploy.py --domain finance

To create a model, the caller passes an execution role to SageMaker. The caller
therefore needs iam:PassRole for sagemaker.amazonaws.com. The Studio execution
role has this through AmazonSageMakerFullAccess. This script runs from Positron
or from a workstation, and Positron makes the better demonstration.

There is no SageMaker training job. These datasets fit a logistic regression in
under a second. A training job adds 4 to 8 minutes of preparation and changes
nothing that the audience sees, because SageMaker still hosts the model. To
train and read the scorecard without creating a billable endpoint, pass
--no-deploy.

Each fit becomes one run on the SageMaker managed MLflow tracking server, so
the MLflow UI shows a comparison. If that server is stopped or absent, training
continues and prints the reason. To skip tracking, pass --no-mlflow.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from ml.tracking import Tracker  # noqa: E402

ENTRYPOINT = REPO_ROOT / "ml" / "entrypoint" / "inference.py"

# One query for each demo. Each returns one row for every loan or subject.
TRAINING_QUERY = {
    "finance": """
        SELECT f.loan_id,
               f.fico_at_origination,
               f.dti,
               f.apr,
               f.term_months,
               CAST(f.principal_amount AS double) AS principal_amount,
               f.employment_length_years,
               f.purpose,
               f.charged_off
        FROM fct_loan_performance f
    """,
    # Group the visit history first, so that every row is one subject. This
    # order keeps the subjects who left the study early in the training data.
    "lifesci": """
        WITH early_window AS (
            -- Screening to Week 12 only. A count over the whole study
            -- depends on exposure. A subject who leaves early has fewer
            -- visits available to miss, and that inverts the coefficient.
            -- A fixed early window and a rate keep the feature forward
            -- looking, and keep the sign correct.
            SELECT subject_id,
                   count(*)                                            AS early_visits,
                   CAST(sum(CASE WHEN NOT visit_completed THEN 1 ELSE 0 END) AS double)
                       / count(*)                                      AS missed_visit_rate
            FROM fct_visit_observations
            WHERE visit_id <= 5
            GROUP BY subject_id
        )
        SELECT s.subject_id,
               s.age,
               s.ecog_status,
               s.bmi,
               s.prior_therapy_lines,
               s.baseline_das,
               s.early_ae_count,
               v.missed_visit_rate,
               s.arm,
               s.discontinued
        FROM dim_subject s
        JOIN early_window v ON v.subject_id = s.subject_id
    """,
}

# The categorical columns, which become one-hot. Every other column is
# numeric, and gets standardized.
CATEGORICAL = {"finance": ["purpose"], "lifesci": ["arm"]}

# A stable sort key. Athena gives no guarantee of row order, so the query
# selects this column, the frame is sorted by it, and then it is dropped. The
# train and test split, and the AUC, are the same on every run.
ID_COLUMN = {"finance": "loan_id", "lifesci": "subject_id"}


def read_training_frame(domain: config.Domain, session: boto3.Session) -> pd.DataFrame:
    df = wr.athena.read_sql_query(
        TRAINING_QUERY[domain.key],
        database=domain.database,
        workgroup=config.ATHENA_WORKGROUP,
        s3_output=config.athena_staging(),
        boto3_session=session,
        # ctas_approach=False: the default creates temporary Glue tables, which
        # needs Glue write permission the read-only demo role does not have.
        ctas_approach=False,
    )
    key = ID_COLUMN[domain.key]
    return df.sort_values(key).drop(columns=[key]).reset_index(drop=True).dropna()


# Each feature set becomes one MLflow run, so the tracking server shows a real
# comparison. The sets answer a question that the business asks, and not one
# about a tuning knob: what do the extra features buy?
#
# A grid over the regularization strength was the first design. It gave the
# same AUC to four decimal places for every value, because there are 37,500
# rows and 13 features. Four identical runs demonstrate nothing.
FEATURE_SETS = {
    "finance": {
        # Does the purpose of the loan, and its size and term, add anything
        # over pure credit quality?
        "credit-only": ["fico_at_origination", "dti", "apr"],
        "credit-and-terms": ["fico_at_origination", "dti", "apr", "term_months",
                             "principal_amount", "employment_length_years"],
        "all-features": None,   # None means every column
    },
    "lifesci": {
        # Do the early warning signals from the first 12 weeks add anything
        # over what a site knows at enrollment?
        "baseline-only": ["age", "ecog_status", "bmi", "prior_therapy_lines",
                          "baseline_das", "arm"],
        "early-warning-only": ["early_ae_count", "missed_visit_rate", "arm"],
        "all-features": None,
    },
}

REGULARIZATION_C = 1.0


def build_design(df: pd.DataFrame, domain: config.Domain, keep: list[str] | None = None):
    """Standardize the numeric columns and one-hot the categorical ones."""
    if keep is not None:
        df = df[[c for c in df.columns if c in keep or c == domain.target]]
    categorical = [c for c in CATEGORICAL[domain.key] if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical + [domain.target]]

    numeric_frame = df[numeric].astype(float)
    means = numeric_frame.mean()
    stds = numeric_frame.std().replace(0.0, 1.0)
    standardized = (numeric_frame - means) / stds

    levels = {c: sorted(df[c].dropna().unique().tolist()) for c in categorical}
    parts = [standardized]
    if levels:
        parts.append(pd.concat(
            [
                pd.Series((df[c] == level).astype(float), name=f"{c}={level}")
                for c, ls in levels.items()
                for level in ls
            ],
            axis=1,
        ))
    design = pd.concat(parts, axis=1)
    return design, numeric, means, stds, levels


def scorecard_from(model, design, numeric, means, stds, levels, domain, metrics) -> dict:
    feature_order = design.columns.tolist()
    return {
        "domain": domain.key,
        "company": domain.company,
        "target": domain.target,
        "numeric_features": numeric,
        "categorical": levels,
        "feature_order": feature_order,
        "means": {k: float(v) for k, v in means.items()},
        "stds": {k: float(v) for k, v in stds.items()},
        "coefficients": {
            name: float(w) for name, w in zip(feature_order, model.coef_[0])
        },
        "intercept": float(model.intercept_[0]),
        "metrics": metrics,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def fit_scorecard(df: pd.DataFrame, domain: config.Domain, tracker=None) -> dict:
    """Fit one model for each feature set, and return the best scorecard.

    If a tracker is given, each fit becomes one MLflow run.
    """
    y = df[domain.target].astype(int).to_numpy()
    best = None
    print(f"  {'feature set':<20} {'features':>8}  {'train AUC':>9}  {'test AUC':>8}")

    for name, keep in FEATURE_SETS[domain.key].items():
        design, numeric, means, stds, levels = build_design(df, domain, keep)
        x_train, x_test, y_train, y_test = train_test_split(
            design.to_numpy(), y, test_size=0.25, random_state=20260821, stratify=y
        )
        model = LogisticRegression(max_iter=2000, C=REGULARIZATION_C)
        model.fit(x_train, y_train)
        train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

        metrics = {
            "test_auc": round(float(test_auc), 4),
            "train_auc": round(float(train_auc), 4),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "base_rate": round(float(y.mean()), 4),
            "n_features": int(design.shape[1]),
            "feature_set": name,
        }
        card = scorecard_from(model, design, numeric, means, stds, levels, domain, metrics)
        print(f"  {name:<20} {design.shape[1]:>8}  {train_auc:>9.4f}  {test_auc:>8.4f}")

        if tracker is not None:
            tracker.log_run(domain, name, metrics, card, design.columns.tolist())

        if best is None or test_auc > best["metrics"]["test_auc"]:
            best = card

    print(f"  best: {best['metrics']['feature_set']}, "
          f"test AUC {best['metrics']['test_auc']}")
    if tracker is not None:
        tracker.log_best(domain, best)
    return best


def build_model_tarball(scorecard: dict) -> bytes:
    """Build model.tar.gz, with scorecard.json at the root and code/inference.py.

    This layout is the documented interface of the scikit-learn serving
    container. SAGEMAKER_SUBMIT_DIRECTORY points to /opt/ml/model/code.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = json.dumps(scorecard, indent=2).encode()
        info = tarfile.TarInfo("scorecard.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

        entry = ENTRYPOINT.read_bytes()
        info = tarfile.TarInfo("code/inference.py")
        info.size = len(entry)
        tar.addfile(info, io.BytesIO(entry))
    return buffer.getvalue()


def deploy(scorecard: dict, domain: config.Domain, session: boto3.Session, role_arn: str) -> None:
    sm = session.client("sagemaker")
    stamp = time.strftime("%Y%m%d%H%M%S")
    model_name = f"{domain.endpoint}-{stamp}"

    key = f"{domain.database}/ml/model/{stamp}/model.tar.gz"
    session.client("s3").put_object(
        Bucket=config.bucket(), Key=key, Body=build_model_tarball(scorecard)
    )
    model_data = f"s3://{config.bucket()}/{key}"
    print(f"  artifact  {model_data}")

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": config.serving_image(),
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": config.REGION,
            },
        },
    )
    print(f"  model     {model_name}")

    sm.create_endpoint_config(
        EndpointConfigName=model_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": config.ENDPOINT_INSTANCE_TYPE,
            }
        ],
    )

    existing = [
        e["EndpointName"]
        for e in sm.list_endpoints(NameContains=domain.endpoint)["Endpoints"]
        if e["EndpointName"] == domain.endpoint
    ]
    if existing:
        print(f"  updating existing endpoint {domain.endpoint}")
        sm.update_endpoint(EndpointName=domain.endpoint, EndpointConfigName=model_name)
    else:
        print(f"  creating endpoint {domain.endpoint} ({config.ENDPOINT_INSTANCE_TYPE})")
        sm.create_endpoint(EndpointName=domain.endpoint, EndpointConfigName=model_name)

    _wait_in_service(sm, domain.endpoint, domain.key)


def _wait_in_service(sm, endpoint: str, domain_key: str, timeout_s: int = 1_200) -> None:
    """Wait for InService, and report the status each time it changes.

    This function does not use the boto3 waiter. If a container does not start,
    the waiter stays silent for up to twenty minutes, and then reports an error
    with no cause. This loop shows FailureReason as soon as AWS sets it.
    """
    print(f"  waiting for {endpoint} (typically 5-8 minutes)")
    deadline = time.time() + timeout_s
    last_status = None

    while time.time() < deadline:
        described = sm.describe_endpoint(EndpointName=endpoint)
        status = described["EndpointStatus"]
        if status != last_status:
            print(f"    {time.strftime('%H:%M:%S')}  {status}")
            last_status = status

        if status == "InService":
            print(f"  endpoint {endpoint} is InService")
            return
        if status in ("Failed", "OutOfService", "RollingBack"):
            reason = described.get("FailureReason", "(no FailureReason reported)")
            raise SystemExit(
                f"\nendpoint {endpoint} entered {status}\n"
                f"  reason: {reason}\n"
                f"  logs:   /aws/sagemaker/Endpoints/{endpoint} in CloudWatch\n"
                f"  then:   uv run python ml/teardown.py --domain {domain_key}"
            )
        time.sleep(15)

    raise SystemExit(
        f"\nendpoint {endpoint} did not reach InService within {timeout_s // 60} minutes "
        f"(last status: {last_status}). Check CloudWatch logs at "
        f"/aws/sagemaker/Endpoints/{endpoint}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(config.DOMAINS))
    parser.add_argument(
        "--role-arn",
        default=None,
        help="execution role to pass to SageMaker (default: the caller's own role)",
    )
    parser.add_argument("--no-deploy", action="store_true", help="train only, create nothing")
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="skip experiment tracking, even when the server is running",
    )
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    session = boto3.Session(region_name=config.REGION)

    print(f"{domain.company}: predicting {domain.target}")
    df = read_training_frame(domain, session)
    print(f"  training frame  {len(df):,} rows x {len(df.columns)} cols")

    tracker = None if args.no_mlflow else Tracker.connect()
    scorecard = fit_scorecard(df, domain, tracker)
    m = scorecard["metrics"]
    print(f"\n  test AUC {m['test_auc']}  base rate {m['base_rate']}  n_train {m['n_train']:,}")
    print("\n  strongest coefficients (standardized log-odds):")
    ranked = sorted(scorecard["coefficients"].items(), key=lambda kv: -abs(kv[1]))
    for name, weight in ranked[:8]:
        print(f"    {name:34} {weight:+.3f}")

    local = REPO_ROOT / "ml" / f"scorecard-{domain.key}.json"
    local.write_text(json.dumps(scorecard, indent=2))
    print(f"\n  scorecard written to {local.relative_to(REPO_ROOT)}")

    if args.no_deploy:
        print("\n--no-deploy was set. Nothing was created in AWS.")
        return

    role_arn = args.role_arn or _caller_role_arn(session)
    print(f"\ndeploying with role {role_arn}")
    deploy(scorecard, domain, session, role_arn)
    print(f"\nteardown when finished: uv run python ml/teardown.py --domain {domain.key}")


def _caller_role_arn(session: boto3.Session) -> str:
    """Convert an assumed-role identity into the ARN of the role."""
    arn = session.client("sts").get_caller_identity()["Arn"]
    if ":assumed-role/" in arn:
        account = arn.split(":")[4]
        role = arn.split(":assumed-role/")[1].split("/")[0]
        return f"arn:aws:iam::{account}:role/{role}"
    return arn


if __name__ == "__main__":
    main()
