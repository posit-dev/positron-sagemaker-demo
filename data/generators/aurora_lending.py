"""Synthetic consumer-lending data for the fictional Aurora Lending Group.

Business realism lives here as declarative constants. The charge-off outcome is
generated from an explicit log-odds model so that a logistic regression trained
downstream recovers a recognisable, interpretable scorecard -- which is how
credit risk is actually modelled in this industry.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

def _norm(weights: list[float]) -> np.ndarray:
    """Weights as a probability vector. Lets the constants below stay readable
    without every list having to sum to exactly 1.0."""
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


SEED = 20260821

START_DATE = date(2019, 1, 1)
END_DATE = date(2025, 12, 31)

N_BORROWERS = 20_000
N_LOANS = 50_000

# Loan products: (product_id, name, term_months, purpose, base_apr)
PRODUCTS = [
    (1, "Personal Loan 36", 36, "debt_consolidation", 0.1190),
    (2, "Personal Loan 60", 60, "debt_consolidation", 0.1425),
    (3, "Home Improvement 60", 60, "home_improvement", 0.1075),
    (4, "Auto Refinance 48", 48, "auto_refinance", 0.0895),
    (5, "Medical Expense 24", 24, "medical", 0.1310),
    (6, "Small Business 60", 60, "small_business", 0.1540),
    (7, "Education 84", 84, "education", 0.0985),
    (8, "Credit Card Refi 36", 36, "credit_card_refi", 0.1265),
]
# Relative origination volume per product.
PRODUCT_WEIGHTS = _norm([0.24, 0.18, 0.11, 0.13, 0.07, 0.06, 0.05, 0.16])

STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI", "CO", "MN",
]
STATE_WEIGHTS = _norm(
    [0.12, 0.09, 0.07, 0.07, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04,
     0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
)

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_BAND_WEIGHTS = _norm([0.08, 0.28, 0.24, 0.19, 0.14, 0.07])

INCOME_BANDS = ["<40k", "40-60k", "60-85k", "85-120k", "120-175k", "175k+"]
INCOME_BAND_WEIGHTS = _norm([0.14, 0.22, 0.24, 0.20, 0.13, 0.07])
# Midpoint used to derive a plausible DTI.
INCOME_MIDPOINT = {
    "<40k": 32_000, "40-60k": 50_000, "60-85k": 72_000,
    "85-120k": 100_000, "120-175k": 145_000, "175k+": 225_000,
}

HOMEOWNERSHIP = ["rent", "mortgage", "own"]
HOMEOWNERSHIP_WEIGHTS = _norm([0.42, 0.45, 0.13])

# Aurora's credit policy declines applicants below 580, so no originated loan
# sits under that floor. This keeps the tail of the book plausible -- real
# unsecured lenders do not book paper that charges off at 60%+.
FICO_BANDS = [(580, 620), (620, 660), (660, 700),
              (700, 740), (740, 780), (780, 850)]
FICO_BAND_LABELS = ["580-619", "620-659", "660-699",
                    "700-739", "740-779", "780-850"]
FICO_BAND_WEIGHTS = _norm([0.08, 0.14, 0.21, 0.24, 0.19, 0.14])

# --- Charge-off log-odds model ---------------------------------------------
# Intercept is tuned to land the portfolio charge-off rate around 9-10%.
CHARGEOFF_INTERCEPT = -2.62
CHARGEOFF_BETAS = {
    "fico_z": -0.62,      # better credit -> far less likely to charge off
    "dti_z": 0.47,        # more leverage -> more risk
    # APR is deliberately weak: it is set by risk-based pricing off FICO, so a
    # large coefficient here would double-count the same underlying risk.
    "apr_z": 0.16,
    "term_z": 0.22,       # longer exposure -> more risk
    "employment_z": -0.19,
    "principal_z": 0.14,
}
# Purpose adds a risk premium on top of the continuous drivers.
PURPOSE_EFFECT = {
    "debt_consolidation": 0.16,
    "credit_card_refi": 0.11,
    "home_improvement": -0.14,
    "auto_refinance": -0.22,
    "medical": 0.20,
    "small_business": 0.42,
    "education": -0.06,
}


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std()


def build_dim_date() -> pd.DataFrame:
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    return pd.DataFrame(
        {
            "date_key": days.strftime("%Y%m%d").astype("int32"),
            "full_date": days.date,
            "year": days.year.astype("int16"),
            "quarter": days.quarter.astype("int8"),
            "month": days.month.astype("int8"),
            "month_name": days.strftime("%B"),
            "day_of_month": days.day.astype("int8"),
            "day_of_week": days.dayofweek.astype("int8"),
            "day_name": days.strftime("%A"),
            "is_weekend": days.dayofweek.isin([5, 6]),
        }
    )


def build_dim_loan_product() -> pd.DataFrame:
    return pd.DataFrame(
        PRODUCTS,
        columns=["product_id", "product_name", "term_months", "purpose", "base_apr"],
    ).astype({"product_id": "int16", "term_months": "int16"})


def build_dim_borrower(rng: np.random.Generator) -> pd.DataFrame:
    n = N_BORROWERS
    fico_band_idx = rng.choice(len(FICO_BANDS), size=n, p=FICO_BAND_WEIGHTS)
    lo = np.array([b[0] for b in FICO_BANDS])[fico_band_idx]
    hi = np.array([b[1] for b in FICO_BANDS])[fico_band_idx]
    fico = rng.integers(lo, hi)

    income_band = rng.choice(INCOME_BANDS, size=n, p=INCOME_BAND_WEIGHTS)
    # Employment length correlates mildly with age band.
    age_band = rng.choice(AGE_BANDS, size=n, p=AGE_BAND_WEIGHTS)
    age_floor = np.array([int(b[:2]) for b in age_band])
    employment = np.clip(
        rng.gamma(shape=2.0, scale=3.0, size=n) + (age_floor - 18) * 0.18,
        0, 40,
    ).round(1)

    return pd.DataFrame(
        {
            "borrower_id": np.arange(1, n + 1, dtype="int32"),
            "age_band": age_band,
            "state": rng.choice(STATES, size=n, p=STATE_WEIGHTS),
            "income_band": income_band,
            "annual_income": np.array([INCOME_MIDPOINT[b] for b in income_band])
            * rng.normal(1.0, 0.12, size=n).clip(0.7, 1.4),
            "employment_length_years": employment,
            "credit_score_band": np.array(FICO_BAND_LABELS)[fico_band_idx],
            "fico_score": fico.astype("int16"),
            "homeownership": rng.choice(HOMEOWNERSHIP, size=n, p=HOMEOWNERSHIP_WEIGHTS),
        }
    )


def build_fct_loan_performance(
    rng: np.random.Generator, borrowers: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    n = N_LOANS
    borrower_idx = rng.integers(0, len(borrowers), size=n)
    product_idx = rng.choice(len(products), size=n, p=PRODUCT_WEIGHTS)

    bor = borrowers.iloc[borrower_idx].reset_index(drop=True)
    prod = products.iloc[product_idx].reset_index(drop=True)

    # Originations grow ~12%/yr; sample dates weighted toward recent years.
    span = (END_DATE - START_DATE).days
    year_weight = rng.random(n) ** 0.72  # skew toward 1.0 == recent
    origination = pd.to_datetime(START_DATE) + pd.to_timedelta(
        (year_weight * span).astype(int), unit="D"
    )

    # Principal scales with income, bounded to plausible unsecured limits.
    principal = np.clip(
        bor["annual_income"].to_numpy() * rng.uniform(0.08, 0.42, size=n),
        2_000, 60_000,
    ).round(2)

    # Risk-based pricing: worse FICO pays more than the product's base APR.
    fico = bor["fico_score"].to_numpy()
    apr = (
        prod["base_apr"].to_numpy()
        + (720 - fico) / 720 * 0.085
        + rng.normal(0, 0.006, size=n)
    ).clip(0.0499, 0.3499).round(4)

    monthly_income = bor["annual_income"].to_numpy() / 12
    term = prod["term_months"].to_numpy()
    monthly_payment = principal * (apr / 12) / (1 - (1 + apr / 12) ** -term)
    # Existing obligations plus this loan.
    dti = (
        (rng.uniform(0.08, 0.34, size=n) * monthly_income + monthly_payment)
        / monthly_income
    ).clip(0.03, 0.75).round(4)

    employment = bor["employment_length_years"].to_numpy()
    purpose_premium = prod["purpose"].map(PURPOSE_EFFECT).to_numpy()

    log_odds = (
        CHARGEOFF_INTERCEPT
        + CHARGEOFF_BETAS["fico_z"] * _zscore(fico.astype(float))
        + CHARGEOFF_BETAS["dti_z"] * _zscore(dti)
        + CHARGEOFF_BETAS["apr_z"] * _zscore(apr)
        + CHARGEOFF_BETAS["term_z"] * _zscore(term.astype(float))
        + CHARGEOFF_BETAS["employment_z"] * _zscore(employment)
        + CHARGEOFF_BETAS["principal_z"] * _zscore(principal)
        + purpose_premium
    )
    charged_off = rng.random(n) < 1 / (1 + np.exp(-log_odds))

    # Seasoning: how long the loan has been on book, capped by its term.
    as_of = pd.to_datetime(END_DATE)
    months_elapsed = ((as_of - origination).days / 30.44).astype(int)
    months_on_book = np.minimum(months_elapsed, term)

    # Charged-off loans carry delinquency; current loans mostly do not.
    delinquency = np.where(
        charged_off,
        rng.integers(120, 271, size=n),
        np.where(rng.random(n) < 0.06, rng.integers(1, 90, size=n), 0),
    ).astype("int16")

    paid_fraction = np.clip(months_on_book / term, 0, 1)
    outstanding = np.where(
        charged_off,
        principal * (1 - paid_fraction * 0.45),
        principal * (1 - paid_fraction),
    ).clip(0).round(2)

    return pd.DataFrame(
        {
            "loan_id": np.arange(1, n + 1, dtype="int32"),
            "borrower_id": bor["borrower_id"].to_numpy(),
            "product_id": prod["product_id"].to_numpy(),
            "origination_date_key": origination.strftime("%Y%m%d").astype("int32"),
            "origination_date": origination.date,
            "principal_amount": principal,
            "apr": apr,
            "fico_at_origination": fico,
            "dti": dti,
            "term_months": term,
            "purpose": prod["purpose"].to_numpy(),
            "employment_length_years": employment,
            "months_on_book": months_on_book.astype("int16"),
            "delinquency_days": delinquency,
            "outstanding_balance": outstanding,
            "charged_off": charged_off,
        }
    )


def build_all(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    borrowers = build_dim_borrower(rng)
    products = build_dim_loan_product()
    return {
        "dim_date": build_dim_date(),
        "dim_loan_product": products,
        "dim_borrower": borrowers,
        "fct_loan_performance": build_fct_loan_performance(rng, borrowers, products),
    }
