"""Synthetic Phase III trial data for the fictional Helix Therapeutics.

Study HLX-301 is a randomized, placebo-controlled trial of a new drug in
moderate to severe disease. The efficacy measure is a disease activity score,
or DAS.

The model predicts early discontinuation. This is a subject-level outcome, so
it lives on ``dim_subject`` and not on the fact table. The walkthrough groups
the visit history first, then joins it to the subject. A study team works the
same way, and the query is more honest than a read from one flat table.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _norm(weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


SEED = 20260821

FIRST_ACTIVATION = date(2023, 3, 1)
ENROLLMENT_START = date(2023, 6, 1)
ENROLLMENT_END = date(2024, 9, 30)

N_SUBJECTS = 1_200
N_SITES = 45

# Visit schedule: (visit_id, visit_name, scheduled_day)
VISITS = [
    (1, "Screening", -14),
    (2, "Baseline", 0),
    (3, "Week 4", 28),
    (4, "Week 8", 56),
    (5, "Week 12", 84),
    (6, "Week 16", 112),
    (7, "Week 24", 168),
    (8, "Week 32", 224),
    (9, "Week 40", 280),
    (10, "Week 48", 336),
    (11, "Week 60", 420),
    (12, "End of Study", 448),
]

# region -> (countries, the share of sites in that region)
REGIONS = {
    "North America": (["United States", "Canada"], 0.34),
    "Western Europe": (["Germany", "France", "Spain", "United Kingdom", "Italy"], 0.30),
    "Eastern Europe": (["Poland", "Hungary", "Czechia"], 0.18),
    "Asia Pacific": (["Japan", "South Korea", "Australia"], 0.18),
}

ARMS = ["Treatment", "Placebo"]

ECOG_LEVELS = [0, 1, 2]
ECOG_WEIGHTS = _norm([0.42, 0.45, 0.13])

SEX = ["Female", "Male"]
SEX_WEIGHTS = _norm([0.54, 0.46])

INVESTIGATOR_SURNAMES = [
    "Okafor", "Lindqvist", "Moreau", "Bianchi", "Nakamura", "Kowalski",
    "Ferreira", "Haddad", "Novak", "Andersen", "Reyes", "Sharma",
    "Petrov", "Larsen", "Dubois", "Costa", "Yamamoto", "Varga",
    "Mbeki", "Sorensen", "Rossi", "Nguyen", "Kaufmann", "Silva",
    "Ivanova", "Meyer", "Tanaka", "Zielinski", "Lopez", "Fischer",
]

# --- The discontinuation log-odds model ------------------------------------
# These values set the early discontinuation rate near 15%.
DISCONT_INTERCEPT = -2.05
DISCONT_BETAS = {
    "ae_count_z": 0.71,        # adverse events are the strongest driver
    "ecog_z": 0.34,            # a worse status, less able to continue
    "age_z": 0.26,
    "baseline_das_z": 0.29,    # more disease at the start, more dropout
    "prior_lines_z": 0.22,
    "bmi_z": 0.08,
}
# On the treatment arm, adverse events cause more withdrawals. On placebo, a
# lack of effect causes more. The totals are therefore close to each other.
ARM_EFFECT = {"Treatment": 0.06, "Placebo": -0.06}


def _zscore(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return (v - v.mean()) / v.std()


def build_dim_visit() -> pd.DataFrame:
    df = pd.DataFrame(VISITS, columns=["visit_id", "visit_name", "scheduled_day"])
    df["is_efficacy_visit"] = df["scheduled_day"] >= 0
    return df.astype({"visit_id": "int16", "scheduled_day": "int16"})


def build_dim_site(rng: np.random.Generator) -> pd.DataFrame:
    region_names = list(REGIONS)
    region_share = _norm([REGIONS[r][1] for r in region_names])
    regions = rng.choice(region_names, size=N_SITES, p=region_share)
    countries = [rng.choice(REGIONS[r][0]) for r in regions]

    # A retention effect for each site. Some sites keep subjects better.
    retention_effect = rng.normal(0, 0.28, size=N_SITES).round(3)

    activation_offset = rng.integers(0, 210, size=N_SITES)
    activation = pd.to_datetime(FIRST_ACTIVATION) + pd.to_timedelta(
        activation_offset, unit="D"
    )

    return pd.DataFrame(
        {
            "site_id": np.arange(101, 101 + N_SITES, dtype="int16"),
            "site_name": [
                f"{c} Clinical Research Unit {i:02d}"
                for i, c in enumerate(countries, start=1)
            ],
            "country": countries,
            "region": regions,
            "investigator": [
                f"Dr. {INVESTIGATOR_SURNAMES[i % len(INVESTIGATOR_SURNAMES)]}"
                for i in range(N_SITES)
            ],
            "activation_date": activation.date,
            "site_retention_effect": retention_effect,
        }
    )


def build_dim_subject(rng: np.random.Generator, sites: pd.DataFrame) -> pd.DataFrame:
    n = N_SUBJECTS
    site_idx = rng.integers(0, len(sites), size=n)
    site = sites.iloc[site_idx].reset_index(drop=True)

    age = np.clip(rng.normal(58, 12, size=n), 18, 85).round(0).astype("int16")
    sex = rng.choice(SEX, size=n, p=SEX_WEIGHTS)
    arm = rng.choice(ARMS, size=n)  # 1:1 randomization
    ecog = rng.choice(ECOG_LEVELS, size=n, p=ECOG_WEIGHTS).astype("int8")
    bmi = np.clip(rng.normal(27.4, 4.8, size=n), 16, 48).round(1)
    prior_lines = rng.choice([0, 1, 2, 3], size=n, p=_norm([0.31, 0.37, 0.22, 0.10])).astype("int8")

    # Baseline disease activity score, 0-10 (higher = worse).
    baseline_das = np.clip(
        rng.normal(6.4, 1.15, size=n) + ecog * 0.42 + prior_lines * 0.18,
        1.0, 10.0,
    ).round(2)

    # A subject cannot enroll before the site opened.
    span = (ENROLLMENT_END - ENROLLMENT_START).days
    wanted = pd.to_datetime(ENROLLMENT_START) + pd.to_timedelta(
        rng.integers(0, span + 1, size=n), unit="D"
    )
    activation = pd.to_datetime(site["activation_date"])
    enrollment = pd.Series(np.maximum(wanted.values, activation.values))

    return pd.DataFrame(
        {
            "subject_id": [f"S{i:05d}" for i in range(1, n + 1)],
            "site_id": site["site_id"].to_numpy(),
            "arm": arm,
            "age": age,
            "sex": sex,
            "ecog_status": ecog,
            "bmi": bmi,
            "prior_therapy_lines": prior_lines,
            "baseline_das": baseline_das,
            "enrollment_date": enrollment.dt.date,
            "enrollment_date_key": enrollment.dt.strftime("%Y%m%d").astype("int32"),
            "_site_retention": site["site_retention_effect"].to_numpy(),
        }
    )


def build_fct_visit_observations(
    rng: np.random.Generator, subjects: pd.DataFrame, visits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the fact table, and dim_subject with the outcome added.

    This function makes the adverse events first, because the count of early
    adverse events drives the decision to leave the study. The order in the
    data therefore matches the order of cause and effect in a real trial.
    """
    n = len(subjects)
    n_visits = len(visits)

    arm_is_treatment = (subjects["arm"] == "Treatment").to_numpy()
    baseline_das = subjects["baseline_das"].to_numpy()
    ecog = subjects["ecog_status"].to_numpy()

    # The chance of an adverse event at each visit. It is higher on the
    # treatment arm, and higher early in the study.
    visit_days = visits["scheduled_day"].to_numpy()
    recency = np.exp(-np.maximum(visit_days, 0) / 220.0)  # more events early
    base_ae = np.where(arm_is_treatment, 0.155, 0.085)[:, None] * recency[None, :]
    base_ae = base_ae * (1 + 0.14 * (ecog[:, None] - 1))
    ae_matrix = rng.random((n, n_visits)) < base_ae
    ae_matrix[:, 0] = False  # no AEs recorded at screening

    # The adverse event count to Week 12 is the early warning that a study
    # team can act on.
    early_cols = np.where(visit_days <= 84)[0]
    early_ae_count = ae_matrix[:, early_cols].sum(axis=1)

    log_odds = (
        DISCONT_INTERCEPT
        + DISCONT_BETAS["ae_count_z"] * _zscore(early_ae_count)
        + DISCONT_BETAS["ecog_z"] * _zscore(ecog)
        + DISCONT_BETAS["age_z"] * _zscore(subjects["age"].to_numpy())
        + DISCONT_BETAS["baseline_das_z"] * _zscore(baseline_das)
        + DISCONT_BETAS["prior_lines_z"] * _zscore(subjects["prior_therapy_lines"].to_numpy())
        + DISCONT_BETAS["bmi_z"] * _zscore(subjects["bmi"].to_numpy())
        + pd.Series(subjects["arm"]).map(ARM_EFFECT).to_numpy()
        + subjects["_site_retention"].to_numpy()
    )
    discontinued = rng.random(n) < 1 / (1 + np.exp(-log_odds))

    # A subject who leaves does so after a visit, more often in year one.
    last_visit_idx = np.where(
        discontinued,
        np.clip((rng.beta(1.9, 2.4, size=n) * (n_visits - 2)).astype(int) + 1, 1, n_visits - 2),
        n_visits - 1,
    )

    # The DAS over time. Treatment improves more. A subject who does not
    # respond is more likely to leave.
    treatment_effect = np.where(arm_is_treatment, -2.05, -0.62)
    # A subject who leaves the study responds less well to the drug.
    treatment_effect = treatment_effect + np.where(discontinued, 0.85, -0.16)

    rows = []
    for v_idx in range(n_visits):
        day = visit_days[v_idx]
        active = last_visit_idx >= v_idx
        if not active.any():
            continue
        idx = np.where(active)[0]

        progress = 0.0 if day <= 0 else min(day / 168.0, 1.0)
        das = np.clip(
            baseline_das[idx]
            + treatment_effect[idx] * progress
            + rng.normal(0, 0.55, size=len(idx)),
            0.0, 10.0,
        ).round(2)

        # In a real trial, a visit happens a few days from its protocol day.
        actual_day = day + rng.integers(-3, 6, size=len(idx))

        rows.append(
            pd.DataFrame(
                {
                    "subject_id": subjects["subject_id"].to_numpy()[idx],
                    "visit_id": np.int16(visits["visit_id"].to_numpy()[v_idx]),
                    "site_id": subjects["site_id"].to_numpy()[idx],
                    "study_day": actual_day.astype("int16"),
                    "das_score": das,
                    "systolic_bp": np.clip(rng.normal(128, 13, len(idx)), 85, 200).round(0).astype("int16"),
                    "diastolic_bp": np.clip(rng.normal(78, 9, len(idx)), 50, 120).round(0).astype("int16"),
                    "heart_rate": np.clip(rng.normal(74, 10, len(idx)), 45, 130).round(0).astype("int16"),
                    "weight_kg": np.clip(
                        subjects["bmi"].to_numpy()[idx] * 3.05
                        + rng.normal(0, 1.6, len(idx)), 40, 190,
                    ).round(1),
                    "adverse_event_flag": ae_matrix[idx, v_idx],
                    # A subject attends fewer visits before a formal
                    # withdrawal, so a missed visit is an early warning and not
                    # noise. Over a fixed early window, which the training
                    # query uses, this looks forward and does not leak the
                    # outcome.
                    "visit_completed": rng.random(len(idx))
                    > np.where(discontinued[idx], 0.115, 0.030),
                }
            )
        )

    fact = pd.concat(rows, ignore_index=True)
    # A fixed order, which a person can read.
    fact = fact.sort_values(["subject_id", "visit_id"]).reset_index(drop=True)
    fact.insert(0, "observation_id", np.arange(1, len(fact) + 1, dtype="int32"))

    out_subjects = subjects.drop(columns=["_site_retention"]).copy()
    out_subjects["discontinued"] = discontinued
    out_subjects["last_visit_id"] = visits["visit_id"].to_numpy()[last_visit_idx].astype("int16")
    out_subjects["early_ae_count"] = early_ae_count.astype("int16")

    return fact, out_subjects


def build_all(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sites = build_dim_site(rng)
    visits = build_dim_visit()
    subjects = build_dim_subject(rng, sites)
    fact, subjects = build_fct_visit_observations(rng, subjects, visits)
    return {
        "dim_site": sites.drop(columns=["site_retention_effect"]),
        "dim_visit": visits,
        "dim_subject": subjects,
        "fct_visit_observations": fact,
    }
