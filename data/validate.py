"""Check the generated data before it goes to AWS.

    uv run python data/validate.py --domain finance

This script prints the numbers that a reviewer asks about. If a number is
outside a plausible range, the script exits with an error. A change to a
generator can make the data lose credibility, and this check finds that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

import config  # noqa: E402
from generate import out_dir_for  # noqa: E402

failures: list[str] = []


def check(label: str, value: float, low: float, high: float, fmt: str = ".3f") -> None:
    ok = low <= value <= high
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {label:38} {value:{fmt}}   (expected {low:{fmt}}-{high:{fmt}})")
    if not ok:
        failures.append(f"{label} = {value:{fmt}}, expected {low:{fmt}}-{high:{fmt}}")


def load(domain: config.Domain) -> dict[str, pd.DataFrame]:
    src = out_dir_for(domain, REPO_ROOT / "data")
    if not src.exists():
        raise SystemExit(f"{src} not found. Run data/generate.py --domain {domain.key} first.")
    return {p.name: pd.read_parquet(p / f"{p.name}.parquet") for p in sorted(src.iterdir()) if p.is_dir()}


def validate_finance(t: dict[str, pd.DataFrame]) -> None:
    f, b = t["fct_loan_performance"], t["dim_borrower"]

    print("\nportfolio")
    check("charge-off rate", f["charged_off"].mean(), 0.07, 0.12)
    check("mean APR", f["apr"].mean(), 0.09, 0.16)
    check("mean DTI", f["dti"].mean(), 0.20, 0.40)
    check("mean principal", float(f["principal_amount"].mean()), 12_000, 30_000, ",.0f")
    check("min FICO (credit policy floor 580)", f["fico_at_origination"].min(), 580, 600, ".0f")

    print("\nrisk order: the charge-off rate must fall as FICO rises")
    by_band = f.groupby(f["fico_at_origination"] // 40 * 40)["charged_off"].mean()
    print("   ", " ".join(f"{k}:{v:.3f}" for k, v in by_band.items()))
    worst = by_band.max()
    check("worst FICO-band charge-off rate", worst, 0.12, 0.35)
    monotone = by_band.is_monotonic_decreasing
    print(f"  [{'ok  ' if monotone else 'FAIL'}] charge-off decreasing in FICO       {monotone}")
    if not monotone:
        failures.append("charge-off rate is not monotone decreasing in FICO")

    print("\nreferential integrity")
    orphans = (~f["borrower_id"].isin(b["borrower_id"])).sum()
    check("orphan borrower_id rows", orphans, 0, 0, ".0f")
    check("orphan product_id rows",
          (~f["product_id"].isin(t["dim_loan_product"]["product_id"])).sum(), 0, 0, ".0f")


def validate_lifesci(t: dict[str, pd.DataFrame]) -> None:
    s, f = t["dim_subject"], t["fct_visit_observations"]

    print("\nenrollment and retention")
    check("discontinuation rate", s["discontinued"].mean(), 0.10, 0.22)
    arm_share = s["arm"].value_counts(normalize=True).min()
    check("smaller arm share (1:1 randomization)", arm_share, 0.45, 0.50)
    check("mean age", s["age"].mean(), 50, 66, ".1f")
    check("visits per subject", len(f) / len(s), 8.0, 12.0, ".2f")

    print("\nsafety signal: dropout must rise as early adverse events rise")
    by_ae = s.groupby("early_ae_count")["discontinued"].mean()
    print("   ", " ".join(f"{k}:{v:.3f}" for k, v in by_ae.items()))
    monotone = by_ae.is_monotonic_increasing
    print(f"  [{'ok  ' if monotone else 'FAIL'}] dropout increasing in early AEs      {monotone}")
    if not monotone:
        failures.append("discontinuation is not monotone increasing in early AE count")

    print("\nefficacy signal: treatment DAS must be below placebo by Week 24")
    j = f.merge(s[["subject_id", "arm"]], on="subject_id")
    w24 = j[j["visit_id"] == 7].groupby("arm")["das_score"].mean()
    delta = w24["Placebo"] - w24["Treatment"]
    print(f"    placebo {w24['Placebo']:.2f}  treatment {w24['Treatment']:.2f}")
    check("DAS separation at Week 24", delta, 0.8, 3.0)

    print("\nreferential integrity")
    check("orphan subject_id rows", (~f["subject_id"].isin(s["subject_id"])).sum(), 0, 0, ".0f")
    check("orphan site_id rows", (~s["site_id"].isin(t["dim_site"]["site_id"])).sum(), 0, 0, ".0f")
    check("orphan visit_id rows", (~f["visit_id"].isin(t["dim_visit"]["visit_id"])).sum(), 0, 0, ".0f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(config.DOMAINS))
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    tables = load(domain)

    print(f"{domain.company}: {domain.database}")
    for name, df in tables.items():
        print(f"  {name:26} {len(df):>7,} rows  {len(df.columns):>2} cols")

    (validate_finance if domain.key == "finance" else validate_lifesci)(tables)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
