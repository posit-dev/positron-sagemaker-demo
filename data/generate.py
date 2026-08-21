"""Generate the synthetic dataset for one demo domain.

    uv run python data/generate.py --domain finance
    uv run python data/generate.py --domain lifesci

The two domains are independent demos, not halves of one story, so --domain is
required. Output is parquet under data/synthetic-<database>/<table>/, which is
git-ignored.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

import config  # noqa: E402
from generators import aurora_lending, helix_trials  # noqa: E402

BUILDERS = {"finance": aurora_lending, "lifesci": helix_trials}

# Money columns are stored as parquet decimal128(12,2) so cents survive the
# round-trip into the Glue catalogue exactly. Anything named here is cast at
# write time regardless of which table it appears in.
MONEY_COLUMNS = frozenset(
    {"principal_amount", "outstanding_balance", "annual_income"}
)
MONEY_TYPE = pa.decimal128(12, 2)


def normalize_table(table: pa.Table) -> pa.Table:
    """Pin the on-disk types Athena will infer from.

    Without this, pandas dtypes drive Glue's column types and small surprises
    (money as float, nanosecond timestamps) surface later as query errors.
    """
    for i, name in enumerate(table.schema.names):
        col = table.column(i)
        if name in MONEY_COLUMNS:
            table = table.set_column(i, name, col.cast(MONEY_TYPE))
        elif pa.types.is_timestamp(col.type) and col.type.unit == "ns":
            table = table.set_column(
                i, name, col.cast(pa.timestamp("us", tz=col.type.tz))
            )
    return table


def write_parquet(df: pd.DataFrame, path: Path) -> int:
    table = normalize_table(pa.Table.from_pandas(df, preserve_index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return path.stat().st_size


def out_dir_for(domain: config.Domain, base: Path) -> Path:
    # "synthetic-" prefix signals artificial data at a glance.
    return base / f"synthetic-{domain.database}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--seed", type=int, default=None, help="override the module seed")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data",
        help="parent directory for synthetic-<database>/",
    )
    args = parser.parse_args()

    domain = config.get_domain(args.domain)
    module = BUILDERS[args.domain]
    seed = args.seed if args.seed is not None else module.SEED

    print(f"{domain.company} -- database {domain.database} (seed {seed})")
    started = time.perf_counter()
    tables = module.build_all(seed)

    out_dir = out_dir_for(domain, args.out_dir)
    for name, df in tables.items():
        size = write_parquet(df, out_dir / name / f"{name}.parquet")
        print(f"  {name:26} {len(df):>7,} rows  {len(df.columns):>2} cols  {size/1024:>8,.0f} KiB")

    print(f"wrote {out_dir.relative_to(REPO_ROOT)} in {time.perf_counter()-started:.1f}s")
    print(f"next: uv run python load/load_athena.py --domain {args.domain}")


if __name__ == "__main__":
    main()
