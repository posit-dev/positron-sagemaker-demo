"""Prepare a fresh AWS account for this demo.

    uv run python setup/bootstrap_aws.py --dry-run
    uv run python setup/bootstrap_aws.py

The demo scripts make the S3 bucket, the Glue databases, the SageMaker model and
endpoint, and the MLflow tracking server. They do not make the execution role,
and they cannot make the SageMaker domain. This script closes that gap.

What it does:

  1. reports the account and the region
  2. makes sure the region offers the services the demo needs
  3. creates the execution role, with a trust policy for SageMaker
  4. attaches AmazonSageMakerFullAccess to that role
  5. renders the IAM template for this account and attaches it
  6. tests the result with the IAM policy simulator

You can run this script more than once. It reports what already exists and
changes nothing that is already correct.

What is left for a person afterward: the SageMaker domain, the Positron image,
and a Posit Connect server. The script prints these at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
import botocore

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

DEFAULT_ROLE_NAME = "sagemaker-demo-execution-role"
INLINE_POLICY_NAME = "PositConf2026AthenaAccess"
MANAGED_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
TEMPLATE = REPO_ROOT / "iam" / "sagemaker-demo-athena-access.template.json"

# SageMaker assumes this role, both for an endpoint and for the MLflow server.
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

problems: list[str] = []


def ok(message: str) -> None:
    print(f"  [ok  ] {message}")


def did(message: str) -> None:
    print(f"  [made] {message}")


def bad(message: str) -> None:
    print(f"  [FAIL] {message}")
    problems.append(message)


def note(message: str) -> None:
    print(f"  [note] {message}")


def render_policy(account: str) -> str:
    text = TEMPLATE.read_text()
    text = text.replace("ACCOUNT_ID", account).replace("REGION", config.REGION)
    json.loads(text)  # fail here rather than inside IAM
    return text


def check_region(session: boto3.Session) -> None:
    """Athena and Glue are everywhere. Managed MLflow is not."""
    try:
        session.client("athena").get_work_group(WorkGroup=config.ATHENA_WORKGROUP)
        ok(f"athena workgroup {config.ATHENA_WORKGROUP} is present")
    except botocore.exceptions.ClientError as err:
        bad(f"athena workgroup {config.ATHENA_WORKGROUP} is missing ({err.response['Error']['Code']})")

    try:
        session.client("sagemaker").list_mlflow_tracking_servers(MaxResults=1)
        ok(f"managed MLflow is offered in {config.REGION}")
    except botocore.exceptions.ClientError as err:
        note(f"managed MLflow is not available in {config.REGION} "
             f"({err.response['Error']['Code']}). The demo runs without it, and "
             "section 10 of the life sciences walkthrough is skipped.")

    lf = session.client("lakeformation")
    try:
        settings = lf.get_data_lake_settings()["DataLakeSettings"]
        defaults = settings.get("CreateDatabaseDefaultPermissions", [])
        principals = [d.get("Principal", {}).get("DataLakePrincipalIdentifier")
                      for d in defaults]
        if "IAM_ALLOWED_PRINCIPALS" in principals:
            ok("Lake Formation defers to IAM, so no extra grant is needed")
        else:
            note("Lake Formation does not use IAM_ALLOWED_PRINCIPALS here. "
                 "Grant the role access to the two demo databases, or Athena "
                 "returns an access error that IAM alone cannot explain.")
    except botocore.exceptions.ClientError:
        note("cannot read the Lake Formation settings. If Athena reports an "
             "access error later, look there first.")


def ensure_role(iam, account: str, role_name: str, dry_run: bool) -> str:
    arn = f"arn:aws:iam::{account}:role/{role_name}"
    try:
        current = iam.get_role(RoleName=role_name)["Role"]
        ok(f"role {role_name} exists")
        services = [
            s.get("Principal", {}).get("Service")
            for s in current["AssumeRolePolicyDocument"]["Statement"]
        ]
        flat = [x for s in services for x in ([s] if isinstance(s, str) else s or [])]
        if "sagemaker.amazonaws.com" in flat:
            ok("its trust policy allows sagemaker.amazonaws.com")
        else:
            bad(f"role {role_name} does not trust sagemaker.amazonaws.com. "
                "SageMaker cannot assume it.")
        return arn
    except iam.exceptions.NoSuchEntityException:
        pass

    if dry_run:
        note(f"would create role {role_name}")
        return arn

    iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
        Description="Execution role for the Positron on SageMaker demo.",
        Tags=[{"Key": "project", "Value": "posit-conf-2026-demo"}],
    )
    did(f"created role {role_name}")
    return arn


def ensure_managed_policy(iam, role_name: str, dry_run: bool) -> None:
    attached = [
        p["PolicyArn"]
        for p in iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]
    ]
    if MANAGED_POLICY in attached:
        ok("AmazonSageMakerFullAccess is attached")
        return
    if dry_run:
        note("would attach AmazonSageMakerFullAccess")
        return
    iam.attach_role_policy(RoleName=role_name, PolicyArn=MANAGED_POLICY)
    did("attached AmazonSageMakerFullAccess")


def ensure_inline_policy(iam, account: str, role_name: str, dry_run: bool) -> None:
    wanted = json.loads(render_policy(account))
    try:
        current = iam.get_role_policy(
            RoleName=role_name, PolicyName=INLINE_POLICY_NAME)["PolicyDocument"]
        if current == wanted:
            ok(f"{INLINE_POLICY_NAME} is attached and current")
            return
        message = f"{INLINE_POLICY_NAME} differs from the template"
    except iam.exceptions.NoSuchEntityException:
        message = f"{INLINE_POLICY_NAME} is absent"

    if dry_run:
        note(f"would write {INLINE_POLICY_NAME} ({message})")
        return
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=INLINE_POLICY_NAME,
        PolicyDocument=json.dumps(wanted),
    )
    did(f"wrote {INLINE_POLICY_NAME} ({message})")


def simulate(iam, role_arn: str, account: str) -> None:
    """Test the result rather than trust it."""
    region, acct = config.REGION, account
    checks = [
        ("athena:StartQueryExecution",
         f"arn:aws:athena:{region}:{acct}:workgroup/{config.ATHENA_WORKGROUP}"),
        ("glue:GetTable",
         f"arn:aws:glue:{region}:{acct}:table/aurora_lending/fct_loan_performance"),
        ("s3:PutObject", f"arn:aws:s3:::{config.bucket()}/athena-query-results/x.csv"),
        ("sagemaker:InvokeEndpoint",
         f"arn:aws:sagemaker:{region}:{acct}:endpoint/{config.FINANCE.endpoint}"),
        ("sagemaker-mlflow:CreateRun",
         f"arn:aws:sagemaker:{region}:{acct}"
         f":mlflow-tracking-server/{config.MLFLOW_SERVER_NAME}"),
    ]
    for action, resource in checks:
        try:
            decision = iam.simulate_principal_policy(
                PolicySourceArn=role_arn, ActionNames=[action], ResourceArns=[resource]
            )["EvaluationResults"][0]["EvalDecision"]
        except botocore.exceptions.ClientError as err:
            note(f"cannot simulate {action} ({err.response['Error']['Code']})")
            continue
        (ok if decision == "allowed" else bad)(f"{decision:<12} {action}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, and change nothing")
    parser.add_argument("--role-name", default=DEFAULT_ROLE_NAME)
    args = parser.parse_args()

    role_name = args.role_name

    session = boto3.Session(region_name=config.REGION)
    try:
        identity = session.client("sts").get_caller_identity()
    except Exception as err:  # noqa: BLE001
        raise SystemExit(f"No AWS credentials: {err}")
    account = identity["Account"]

    print(f"account {account} / region {config.REGION}")
    if args.dry_run:
        print("dry run. Nothing will change.\n")
    print()

    print("region")
    check_region(session)

    print("\nexecution role")
    iam = session.client("iam")
    try:
        role_arn = ensure_role(iam, account, role_name, args.dry_run)
        ensure_managed_policy(iam, role_name, args.dry_run)
        ensure_inline_policy(iam, account, role_name, args.dry_run)
    except botocore.exceptions.ClientError as err:
        if err.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException"):
            bad("this identity cannot write IAM. Ask an administrator to run this "
                "script, or to create the role from iam/README-bootstrap.md.")
            raise SystemExit(1) from None
        raise

    if not args.dry_run:
        print("\npermissions, as evaluated by AWS")
        simulate(iam, role_arn, account)

    print("\nstill to do by hand")
    print("  1. Create a SageMaker domain, and set its execution role to")
    print(f"     {role_name}. Use network mode PublicInternetOnly, or add VPC")
    print("     endpoints for sagemaker.api, sagemaker.runtime, athena, glue and S3.")
    print("  2. Build the Positron image and attach it to that domain. Follow the")
    print("     Positron on SageMaker setup guide.")
    print("  3. Register a Posit Connect server, if you want to publish the report.")
    print("\nthen")
    print("  uv run python data/generate.py    --domain lifesci")
    print("  uv run python load/load_athena.py --domain lifesci")
    print("  bash setup/verify-env.sh lifesci")

    if problems:
        print(f"\n{len(problems)} problem(s) above")
        raise SystemExit(1)
    print("\nthe account is ready")


if __name__ == "__main__":
    main()
