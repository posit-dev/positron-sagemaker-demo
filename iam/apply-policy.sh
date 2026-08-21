#!/usr/bin/env bash
# Fill in the account and region, then attach a policy template to a role.
#
#   bash iam/apply-policy.sh sagemaker-demo-athena-access sagemaker-demo-execution-role
#
# The templates hold ACCOUNT_ID and REGION placeholders, because this is a
# public repository and it must not contain an AWS account number. This script
# reads the account from STS.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

TEMPLATE="${1:-}"
ROLE="${2:-}"
POLICY_NAME="${3:-PositConf2026AthenaAccess}"

if [[ -z "$TEMPLATE" || -z "$ROLE" ]]; then
  echo "usage: bash iam/apply-policy.sh <template-name> <role-name> [policy-name]" >&2
  echo "templates:" >&2
  ls iam/*.template.json | sed 's|iam/||; s|\.template\.json||; s|^|  |' >&2
  exit 2
fi

SRC="iam/${TEMPLATE}.template.json"
if [[ ! -f "$SRC" ]]; then
  echo "no such template: $SRC" >&2
  exit 2
fi

REGION="${POSIT_DEMO_REGION:-$(aws configure get region)}"
ACCOUNT_ID="${POSIT_DEMO_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
echo "account $ACCOUNT_ID / region $REGION"

RENDERED="$(mktemp -t posit-demo-policy)"
trap 'rm -f "$RENDERED"' EXIT
sed -e "s/ACCOUNT_ID/${ACCOUNT_ID}/g" -e "s/REGION/${REGION}/g" "$SRC" > "$RENDERED"

python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$RENDERED"

aws iam put-role-policy \
  --role-name "$ROLE" \
  --policy-name "$POLICY_NAME" \
  --policy-document "file://${RENDERED}"

echo "attached $POLICY_NAME to $ROLE"
