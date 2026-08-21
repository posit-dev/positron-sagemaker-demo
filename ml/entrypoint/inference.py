"""SageMaker serving entrypoint for the demo risk models.

Deliberately numpy-only. The model is persisted as a JSON scorecard -- feature
means, standard deviations and logistic coefficients -- rather than a pickled
estimator, so there is no scikit-learn version coupling between the machine that
trained the model and the container that serves it. That coupling is the usual
cause of "works locally, 500s on the endpoint".

Contract:
    POST  application/json   {"instances": [{feature: value, ...}, ...]}
    ->    application/json   {"predictions": [p, ...], "target": "charged_off"}
"""

from __future__ import annotations

import json
import os

import numpy as np

SCORECARD_FILENAME = "scorecard.json"


def model_fn(model_dir: str) -> dict:
    with open(os.path.join(model_dir, SCORECARD_FILENAME)) as fh:
        return json.load(fh)


def input_fn(request_body, request_content_type="application/json"):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)
    # Accept either {"instances": [...]} or a bare list of records.
    instances = payload["instances"] if isinstance(payload, dict) else payload
    if isinstance(instances, dict):
        instances = [instances]
    return instances


def _design_row(record: dict, card: dict) -> np.ndarray:
    """Standardised numeric features followed by one-hot categoricals."""
    values = []
    for name in card["numeric_features"]:
        raw = record.get(name)
        if raw is None:
            raw = card["means"][name]  # mean-impute a missing feature
        std = card["stds"][name] or 1.0
        values.append((float(raw) - card["means"][name]) / std)

    for column, levels in card["categorical"].items():
        observed = record.get(column)
        for level in levels:
            values.append(1.0 if observed == level else 0.0)

    return np.asarray(values, dtype=float)


def predict_fn(instances, card):
    design = np.vstack([_design_row(r, card) for r in instances])
    weights = np.asarray(
        [card["coefficients"][n] for n in card["feature_order"]], dtype=float
    )
    logits = design @ weights + card["intercept"]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return {
        "predictions": [round(float(p), 6) for p in probabilities],
        "target": card["target"],
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
