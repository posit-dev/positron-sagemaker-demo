#!/usr/bin/env bash
# Render one report and publish it to Posit Connect.
#
#   bash setup/publish.sh finance
#   bash setup/publish.sh lifesci  my-connect-server
#
# Each report is a self-contained document. The .qmd file declares the AWS
# settings it needs, and does not import config.py, because the bundle does not
# hold config.py. tests/test_report_config.py keeps the two in agreement.
#
# The Positron SageMaker image does not contain rsconnect-python, so uv runs it
# instead. Quarto runs through `uv run`, so that it uses the project
# environment and not the system Python.
#
# For credentials, either register the server one time:
#   uv tool run --from rsconnect-python rsconnect add \
#     --server https://<your-connect-server> --api-key "$KEY" --name pub
# or export CONNECT_SERVER and CONNECT_API_KEY. Note that --name and
# CONNECT_SERVER are mutually exclusive; passing both is an error.
#
# Connect renders the report against live Athena, so Connect needs AWS
# credentials as content variables. The policy to use is
# iam/connect-reader-policy.template.json.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

DOMAIN="${1:-}"
SERVER_NAME="${2:-pub}"

case "$DOMAIN" in
  finance) QMD=reports/aurora_lending/portfolio_risk_review.qmd
           TITLE="Aurora Lending Group: Portfolio Risk Review" ;;
  lifesci) QMD=reports/helix_trials/enrollment_safety_review.qmd
           TITLE="Helix Therapeutics: Study HLX-301 Enrollment and Retention Review" ;;
  *) echo "usage: bash setup/publish.sh <finance|lifesci> [connect-server-name]" >&2
     exit 2 ;;
esac

DIR="$(dirname "$QMD")"
ENTRY="$(basename "$QMD")"

echo "checking report constants against config.py"
uv run python tests/test_report_config.py >/dev/null

# Copy the brand file next to the report, so that the bundle renders alone.
cp _brand.yml "$DIR/_brand.yml"
cp reports/requirements.txt "$DIR/requirements.txt"

echo "rendering $QMD"
uv run quarto render "$QMD"

echo "publishing to Connect ($SERVER_NAME)"
if [[ -n "${CONNECT_SERVER:-}" ]]; then
  TARGET=(--server "$CONNECT_SERVER" --api-key "${CONNECT_API_KEY:?CONNECT_API_KEY must be set}")
else
  TARGET=(--name "$SERVER_NAME")
fi

uv tool run --from rsconnect-python rsconnect deploy quarto \
  "${TARGET[@]}" \
  --entrypoint "$ENTRY" \
  --title "$TITLE" \
  --exclude '*.html' \
  "$DIR"

echo
echo "If Connect has no AWS credentials, publish the rendered output instead:"
echo "  uv run quarto publish connect \"$QMD\""
