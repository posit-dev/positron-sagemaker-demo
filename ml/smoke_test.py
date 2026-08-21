"""Confirm a deployed endpoint returns sensible scores. Run right after deploy.

    uv run python ml/smoke_test.py --domain finance

Pulls a small sample from Athena, scores it on the endpoint, and checks that the
probabilities separate actual positives from actual negatives. Catches a broken
serving container in about ten seconds, which is much better than discovering it
mid-session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
import botocore
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from ml.train_and_deploy import read_training_frame  # noqa: E402


def invoke(endpoint: str, records: list[dict], session: boto3.Session) -> list[float]:
    response = session.client("sagemaker-runtime").invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps({"instances": records}),
    )
    return json.loads(response["Body"].read())["predictions"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(config.DOMAINS))
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    session = boto3.Session(region_name=config.REGION)

    # Same reader the trainer uses, so the features match exactly.
    df = read_training_frame(domain, session)
    df = df.sample(min(args.n, len(df)), random_state=1)

    actual = df[domain.target].astype(int).to_numpy()
    records = df.drop(columns=[domain.target]).to_dict(orient="records")
    try:
        scores = np.array(invoke(domain.endpoint, records, session))
    except botocore.exceptions.ClientError as err:
        # A missing endpoint is the normal state between sessions, so say what to
        # do about it rather than dumping a traceback.
        raise SystemExit(
            f"Endpoint {domain.endpoint} is not usable: "
            f"{err.response['Error']['Message']}\n"
            f"Deploy it with: uv run python ml/train_and_deploy.py --domain {domain.key}"
        ) from None

    print(f"{domain.company} -- endpoint {domain.endpoint}")
    print(f"  scored {len(scores)} records")
    print(f"  probability range   [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  mean where {domain.target}=1  {scores[actual == 1].mean():.4f}")
    print(f"  mean where {domain.target}=0  {scores[actual == 0].mean():.4f}")

    problems = []
    if not ((scores >= 0).all() and (scores <= 1).all()):
        problems.append("probabilities outside [0, 1]")
    if scores[actual == 1].mean() <= scores[actual == 0].mean():
        problems.append("positives do not score higher than negatives")
    if scores.std() < 1e-6:
        problems.append("all predictions identical -- model or features not wired up")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nendpoint healthy")


if __name__ == "__main__":
    main()
