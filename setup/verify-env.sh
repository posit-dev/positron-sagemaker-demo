#!/usr/bin/env bash
# One check before a session. Run this inside Positron on SageMaker.
#
#   bash setup/verify-env.sh finance
#   bash setup/verify-env.sh lifesci
#
# The script tests the environment, the credentials, the catalog, and the
# endpoint. It then reports which one is not ready.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DOMAIN="${1:-}"
if [[ "$DOMAIN" != "finance" && "$DOMAIN" != "lifesci" ]]; then
  echo "usage: bash setup/verify-env.sh <finance|lifesci>" >&2
  exit 2
fi

fail=0
ok()   { printf '  [ok  ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }

echo "tooling"
command -v uv     >/dev/null && ok "uv     $(uv --version)"        || bad "uv not on PATH"
command -v quarto >/dev/null && ok "quarto $(quarto --version)"    || bad "quarto not on PATH"

echo
echo "python environment"
if uv sync --quiet 2>/dev/null; then
  ok "uv sync clean"
else
  bad "uv sync failed. Run 'uv sync' and read the error."
fi
uv run python -c "import awswrangler, boto3, pandas, sklearn, great_tables" 2>/dev/null \
  && ok "core imports resolve" || bad "core imports failed"

echo
echo "aws"
uv run python - "$DOMAIN" <<'PY'
import sys
sys.path.insert(0, ".")
import boto3, botocore, awswrangler as wr
import config

domain = config.get_domain(sys.argv[1])
session = boto3.Session(region_name=config.REGION)
rc = 0

def ok(m):  print(f"  [ok  ] {m}")
def bad(m):
    global rc
    print(f"  [FAIL] {m}"); rc = 1

try:
    ident = session.client("sts").get_caller_identity()
    ok(f"identity {ident['Arn'].split('/')[-2] if '/' in ident['Arn'] else ident['Arn']}")
except Exception as e:
    bad(f"no AWS credentials ({e})")
    raise SystemExit(1)

# Athena is the only service that needs a policy which the role does not
# already have.
try:
    n = wr.athena.read_sql_query(
        f"SELECT count(*) AS n FROM {domain.fact_table}",
        database=domain.database, workgroup=config.ATHENA_WORKGROUP,
        s3_output=config.athena_staging(), boto3_session=session,
        # Use the same option that the demo uses, so that this check tests
        # the same permissions. The default option needs a Glue write.
        ctas_approach=False,
    )["n"].iloc[0]
    ok(f"athena {domain.database}.{domain.fact_table}  {n:,} rows")
except Exception as e:
    bad(f"athena query failed ({type(e).__name__}). Is the PositConf2026AthenaAccess policy attached?")

try:
    status = session.client("sagemaker").describe_endpoint(
        EndpointName=domain.endpoint)["EndpointStatus"]
    (ok if status == "InService" else bad)(f"endpoint {domain.endpoint} is {status}")
except botocore.exceptions.ClientError:
    bad(f"endpoint {domain.endpoint} does not exist. "
        f"Run: uv run python ml/train_and_deploy.py --domain {domain.key}")

raise SystemExit(rc)
PY
[[ $? -ne 0 ]] && fail=1

echo
if [[ $fail -eq 0 ]]; then
  echo "ready for the $DOMAIN demo"
else
  echo "NOT ready. Correct the [FAIL] lines above."
fi
exit $fail
