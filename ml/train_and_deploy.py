"""Train a risk model from Athena and host it on a SageMaker endpoint.

    uv run python ml/train_and_deploy.py --domain finance

Creating a model passes an execution role to SageMaker, so the caller needs
iam:PassRole for sagemaker.amazonaws.com. The Studio execution role has it via
AmazonSageMakerFullAccess, and this account's SSO permission set does too
(verified) -- so this runs from Positron or from a workstation. Running it from
Positron is the better demo narrative.

There is deliberately no SageMaker training job. The datasets are small enough
to fit a logistic regression in under a second, and skipping the job removes a
4-8 minute wait from demo prep for no loss of realism -- the model is still
hosted by SageMaker, which is the part the audience sees. Pass --no-deploy to
train and inspect the scorecard without creating a billable endpoint.
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

ENTRYPOINT = REPO_ROOT / "ml" / "entrypoint" / "inference.py"

# One query per domain producing a flat, subject/loan-level training frame.
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
    # Subject-level: visit history is aggregated first, so every row is one
    # subject. Aggregating before the join is what keeps subjects who left the
    # study early in the training set.
    "lifesci": """
        WITH early_window AS (
            -- Screening through Week 12 only. A raw count over the whole study
            -- would be confounded by exposure: subjects who leave early have
            -- fewer visits to miss, which inverts the coefficient. Restricting
            -- to a fixed early window and using a rate keeps the feature
            -- prospective and the sign interpretable.
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

# Which columns are categorical (one-hot) rather than numeric (standardised).
CATEGORICAL = {"finance": ["purpose"], "lifesci": ["arm"]}

# Stable sort key, selected only so the row order is deterministic, then dropped
# before fitting. awswrangler's CTAS read gives no ordering guarantee.
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


def fit_scorecard(df: pd.DataFrame, domain: config.Domain) -> dict:
    categorical = CATEGORICAL[domain.key]
    numeric = [c for c in df.columns if c not in categorical + [domain.target]]

    y = df[domain.target].astype(int).to_numpy()
    numeric_frame = df[numeric].astype(float)
    means = numeric_frame.mean()
    stds = numeric_frame.std().replace(0.0, 1.0)
    standardised = (numeric_frame - means) / stds

    levels = {c: sorted(df[c].dropna().unique().tolist()) for c in categorical}
    one_hot = pd.concat(
        [
            pd.Series((df[c] == level).astype(float), name=f"{c}={level}")
            for c, ls in levels.items()
            for level in ls
        ],
        axis=1,
    )

    design = pd.concat([standardised, one_hot], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        design.to_numpy(), y, test_size=0.25, random_state=20260821, stratify=y
    )

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

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
        "metrics": {
            "test_auc": round(float(auc), 4),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "base_rate": round(float(y.mean()), 4),
        },
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_model_tarball(scorecard: dict) -> bytes:
    """model.tar.gz with the scorecard at the root and inference.py under code/.

    That layout is the scikit-learn serving container's documented contract:
    SAGEMAKER_SUBMIT_DIRECTORY points at /opt/ml/model/code.
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
    """Poll for InService, reporting status as it changes.

    Deliberately not boto3's waiter: a container that fails to start would leave
    the waiter silent for up to twenty minutes and then raise without saying why.
    Polling lets us surface FailureReason the moment it appears, which is the
    difference between a thirty-second diagnosis and a twenty-minute one.
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
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    session = boto3.Session(region_name=config.REGION)

    print(f"{domain.company} -- predicting {domain.target}")
    df = read_training_frame(domain, session)
    print(f"  training frame  {len(df):,} rows x {len(df.columns)} cols")

    scorecard = fit_scorecard(df, domain)
    m = scorecard["metrics"]
    print(f"  test AUC {m['test_auc']}  base rate {m['base_rate']}  n_train {m['n_train']:,}")
    print("\n  strongest coefficients (standardised log-odds):")
    ranked = sorted(scorecard["coefficients"].items(), key=lambda kv: -abs(kv[1]))
    for name, weight in ranked[:8]:
        print(f"    {name:34} {weight:+.3f}")

    local = REPO_ROOT / "ml" / f"scorecard-{domain.key}.json"
    local.write_text(json.dumps(scorecard, indent=2))
    print(f"\n  scorecard written to {local.relative_to(REPO_ROOT)}")

    if args.no_deploy:
        print("\n--no-deploy set; nothing created in AWS")
        return

    role_arn = args.role_arn or _caller_role_arn(session)
    print(f"\ndeploying with role {role_arn}")
    deploy(scorecard, domain, session, role_arn)
    print(f"\nteardown when finished: uv run python ml/teardown.py --domain {domain.key}")


def _caller_role_arn(session: boto3.Session) -> str:
    """Turn an assumed-role identity into the underlying role ARN."""
    arn = session.client("sts").get_caller_identity()["Arn"]
    if ":assumed-role/" in arn:
        account = arn.split(":")[4]
        role = arn.split(":assumed-role/")[1].split("/")[0]
        return f"arn:aws:iam::{account}:role/{role}"
    return arn


if __name__ == "__main__":
    main()
