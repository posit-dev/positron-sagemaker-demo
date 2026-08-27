"""Make sure the report settings still agree with config.py.

Each report declares its own AWS settings, so that Posit Connect can render it
as a self-contained document. The alternative is to put the whole repository in
the bundle. This test makes the repeated settings safe.

    uv run python tests/test_report_config.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

REPORTS = {
    config.FINANCE: REPO_ROOT / config.FINANCE.report,
    config.LIFESCI: REPO_ROOT / config.LIFESCI.report,
}


def inlined(text: str, name: str) -> str:
    match = re.search(rf'^{name} = os\.environ\.get\(\s*"[^"]+",\s*"([^"]+)"', text, re.M)
    if not match:
        raise AssertionError(f"{name} not found as an env-var-with-default in the report")
    return match.group(1)


def main() -> int:
    failures = []
    for domain, path in REPORTS.items():
        text = path.read_text()
        expected = {
            "REGION": config.REGION,
            "WORKGROUP": config.ATHENA_WORKGROUP,
            "DATABASE": domain.database,
            "ENDPOINT": domain.endpoint,
            "ATHENA_DRIVER": config.ATHENA_ODBC_DRIVER,
            "ATHENA_AUTH": config.ATHENA_ODBC_AUTH,
        }
        for name, want in expected.items():
            got = inlined(text, name)
            status = "ok  " if got == want else "FAIL"
            print(f"  [{status}] {path.name:34} {name:10} {got}")
            if got != want:
                failures.append(f"{path.name}: {name} is {got!r}, config.py says {want!r}")

        # The script builds the bucket name, so compare the prefix only.
        prefix = re.search(r'^BUCKET_PREFIX = "([^"]+)"', text, re.M)
        if prefix is None:
            failures.append(f"{path.name} has no BUCKET_PREFIX")
        elif prefix.group(1) != config.BUCKET_PREFIX:
            failures.append(
                f"{path.name}: BUCKET_PREFIX is {prefix.group(1)!r}, "
                f"config.py says {config.BUCKET_PREFIX!r}"
            )
        else:
            print(f"  [ok  ] {path.name:34} {'BUCKET_PREFIX':10} {prefix.group(1)}")

        # The report must not import config, which the Connect bundle lacks.
        if re.search(r"^import config", text, re.M):
            failures.append(f"{path.name} imports config, which will not exist on Connect")

        # This repository is public, so it must hold no account number.
        for hit in re.findall(r"\b\d{12}\b", text):
            failures.append(f"{path.name} contains what looks like an AWS account ID: {hit}")

    if failures:
        print("\n" + "\n".join(f"  - {f}" for f in failures))
        return 1
    print("\nreport constants match config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
