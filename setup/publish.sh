#!/usr/bin/env bash
# Render one report and publish it to Posit Connect.
#
#   bash setup/publish.sh finance
#   bash setup/publish.sh lifesci  my-connect-server
#
# The reports are deployed as SELF-CONTAINED documents: each .qmd inlines the
# five AWS settings it needs rather than importing config.py, because config.py
# is not in the bundle. tests/test_report_config.py keeps the two in sync.
#
# rsconnect-python is not baked into the Positron SageMaker image, so it is run
# through uv rather than installed. Quarto is invoked via `uv run` so it picks up
# the project venv instead of the system python.
#
# Credentials -- either register the server once:
#   uv tool run --from rsconnect-python rsconnect add \
#     --server https://<your-connect-server> --api-key "$KEY" --name pub
# or export CONNECT_SERVER and CONNECT_API_KEY. Note that --name and
# CONNECT_SERVER are mutually exclusive; passing both is an error.
#
# Connect re-renders against live Athena, so it needs AWS credentials set as
# content Vars. iam/connect-reader-policy.json is the least-privilege policy.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

DOMAIN="${1:-}"
SERVER_NAME="${2:-pub}"

case "$DOMAIN" in
  finance) QMD=reports/aurora_lending/portfolio_risk_review.qmd
           TITLE="Aurora Lending Group -- Portfolio Risk Review" ;;
  lifesci) QMD=reports/helix_trials/enrollment_safety_review.qmd
           TITLE="Helix Therapeutics -- HLX-301 Enrolment & Retention Review" ;;
  *) echo "usage: bash setup/publish.sh <finance|lifesci> [connect-server-name]" >&2
     exit 2 ;;
esac

DIR="$(dirname "$QMD")"
ENTRY="$(basename "$QMD")"

echo "checking report constants against config.py"
uv run python tests/test_report_config.py >/dev/null

# Stage the brand file beside the report so the bundle renders standalone.
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
