#!/usr/bin/env bash
# Pre-demo check. Run this inside Positron on SageMaker before a session.
#
#   bash setup/verify-env.sh finance
#   bash setup/verify-env.sh lifesci
#
# Checks the environment, the credentials, the catalogue and the endpoint, and
# says plainly which of them is not ready.
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
  bad "uv sync failed -- run 'uv sync' and read the error"
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

# Athena is the one thing that needs a policy the base role does not ship with.
try:
    n = wr.athena.read_sql_query(
        f"SELECT count(*) AS n FROM {domain.fact_table}",
        database=domain.database, workgroup=config.ATHENA_WORKGROUP,
        s3_output=config.athena_staging(), boto3_session=session,
        # Must match how the demo queries, so this check exercises the same
        # permissions: the default CTAS path would need Glue write.
        ctas_approach=False,
    )["n"].iloc[0]
    ok(f"athena {domain.database}.{domain.fact_table}  {n:,} rows")
except Exception as e:
    bad(f"athena query failed -- is the PositConf2026AthenaAccess policy attached? ({type(e).__name__})")

try:
    status = session.client("sagemaker").describe_endpoint(
        EndpointName=domain.endpoint)["EndpointStatus"]
    (ok if status == "InService" else bad)(f"endpoint {domain.endpoint} is {status}")
except botocore.exceptions.ClientError:
    bad(f"endpoint {domain.endpoint} does not exist -- "
        f"run: uv run python ml/train_and_deploy.py --domain {domain.key}")

raise SystemExit(rc)
PY
[[ $? -ne 0 ]] && fail=1

echo
if [[ $fail -eq 0 ]]; then
  echo "ready for the $DOMAIN demo"
else
  echo "NOT ready -- fix the [FAIL] lines above"
fi
exit $fail
