#!/usr/bin/env python3
"""Evaluate λ predictions against observed pairwise experiment metrics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


@dataclass
class PairPrediction:
    base: str
    donor: str
    lambda_lower: Optional[float]
    lambda_upper: Optional[float]
    lambda_estimate: float
    lambda_discrete: float


@dataclass
class PairMetrics:
    base_top1: float
    donor_top1: float


def load_predictions(path: Path) -> Dict[Tuple[str, str], PairPrediction]:
    rows: Dict[Tuple[str, str], PairPrediction] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                base = row["base"]
                donor = row["donor"]
            except KeyError as err:
                raise RuntimeError(f"Missing required column in {path}: {err}") from err
            key = (base, donor)
            lambda_lower = _parse_float(row.get("lambda_lower"))
            lambda_upper = _parse_float(row.get("lambda_upper"))
            prediction = PairPrediction(
                base=base,
                donor=donor,
                lambda_lower=lambda_lower,
                lambda_upper=lambda_upper,
                lambda_estimate=float(row["lambda_estimate"]),
                lambda_discrete=float(row["lambda_discrete"]),
            )
            rows[key] = prediction
    return rows


def detect_percentage_scale(rows: Iterable[csv.DictReader]) -> float:
    max_value = 0.0
    for row in rows:
        try:
            value = abs(float(row["base_top_1_pct"]))
        except (KeyError, ValueError):
            continue
        if value > max_value:
            max_value = value
    return 0.01 if max_value > 1.5 else 1.0


def load_actual_metrics(path: Path) -> Tuple[
    Dict[Tuple[str, str], Dict[float, PairMetrics]],
    float,
]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = list(csv.DictReader(fh))
    scale = detect_percentage_scale(reader)
    data: Dict[Tuple[str, str], Dict[float, PairMetrics]] = {}
    for row in reader:
        try:
            base = row["base"]
            donor = row["donor"]
            lambda_value = float(row["lambda"])
            base_top1 = float(row["base_top_1_pct"]) * scale
            donor_top1 = float(row["donor_top_1_pct"]) * scale
        except (KeyError, ValueError) as err:
            raise RuntimeError(f"Invalid row in {path}: {row}") from err
        key = (base, donor)
        data.setdefault(key, {})[lambda_value] = PairMetrics(
            base_top1=base_top1, donor_top1=donor_top1
        )
    return data, scale


def evaluate_predictions(
    predictions: Dict[Tuple[str, str], PairPrediction],
    actual: Dict[Tuple[str, str], Dict[float, PairMetrics]],
    *,
    base_threshold: float,
    donor_threshold: float,
    per_pair_path: Optional[Path] = None,
) -> None:
    tp = fp = tn = fn = 0
    missing_actual = 0
    missing_lambda = 0

    total_pairs = len(predictions)
    total_loss = 0.0
    total_pairs_with_loss = 0
    within_bounds = 0
    positive_pairs = 0
    detailed_rows: List[Dict[str, object]] = []

    for key, prediction in predictions.items():
        metrics_by_lambda = actual.get(key)
        if metrics_by_lambda is None:
            missing_actual += 1
            continue

        lambda_pred = prediction.lambda_discrete
        if lambda_pred not in metrics_by_lambda:
            missing_lambda += 1
            continue

        pred_metrics = metrics_by_lambda[lambda_pred]
        pred_success = (
            pred_metrics.base_top1 >= base_threshold
            and pred_metrics.donor_top1 <= donor_threshold
        )

        success_lambdas = [
            lam
            for lam, metrics in metrics_by_lambda.items()
            if metrics.base_top1 >= base_threshold
            and metrics.donor_top1 <= donor_threshold
        ]
        actual_success = bool(success_lambdas)
        if actual_success:
            positive_pairs += 1
        if pred_success and actual_success:
            tp += 1
        elif pred_success and not actual_success:
            fp += 1
        elif not pred_success and actual_success:
            fn += 1
        else:
            tn += 1

        base_deficit = max(0.0, base_threshold - pred_metrics.base_top1)
        donor_excess = max(0.0, pred_metrics.donor_top1 - donor_threshold)
        loss = base_deficit + donor_excess
        total_loss += loss
        total_pairs_with_loss += 1

        best_lambda: Optional[float] = min(success_lambdas) if success_lambdas else None
        bounds_ok = False
        if actual_success and best_lambda is not None:
            lower_ok = (
                prediction.lambda_lower is None or best_lambda >= prediction.lambda_lower
            )
            upper_ok = (
                prediction.lambda_upper is None or best_lambda <= prediction.lambda_upper
            )
            bounds_ok = lower_ok and upper_ok
            if bounds_ok:
                within_bounds += 1

        detailed_rows.append(
            {
                "base": prediction.base,
                "donor": prediction.donor,
                "lambda_pred": lambda_pred,
                "base_top1_pred": pred_metrics.base_top1,
                "donor_top1_pred": pred_metrics.donor_top1,
                "pred_success": pred_success,
                "actual_success": actual_success,
                "best_success_lambda": best_lambda,
                "lambda_lower": prediction.lambda_lower,
                "lambda_upper": prediction.lambda_upper,
                "bounds_cover_best": bounds_ok,
                "loss": loss,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    avg_loss = total_loss / total_pairs_with_loss if total_pairs_with_loss else 0.0
    bounds_rate = within_bounds / positive_pairs if positive_pairs else 0.0

    print("λ Prediction Evaluation")
    print("=======================")
    print(f"Total predicted pairs : {total_pairs}")
    print(f"Missing actual rows   : {missing_actual}")
    print(f"Missing λ evaluations : {missing_lambda}")
    print(f"Evaluated pairs       : {total_pairs - missing_actual - missing_lambda}")
    print()
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print()
    print(f"Average loss (base/ donor threshold gap): {avg_loss:.4f}")
    print(f"Positive pairs covered by bounds        : {bounds_rate:.4f}")

    if per_pair_path is not None:
        per_pair_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(detailed_rows[0].keys()) if detailed_rows else []
        with per_pair_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_rows)
        print(f"\nSaved per-pair evaluation to {per_pair_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted λ values against latest pairwise results."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("exp_results/predicted_lambda.csv"),
        help="CSV file containing predicted λ values.",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        default=Path("latest_pairwise.csv"),
        help="CSV file with observed pairwise metrics.",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.1,
        help="Base top-1 threshold treated as acceptable.",
    )
    parser.add_argument(
        "--donor-threshold",
        type=float,
        default=0.001,
        help="Donor top-1 threshold treated as acceptable.",
    )
    parser.add_argument(
        "--per-pair-output",
        type=Path,
        default=None,
        help="Optional path to write per-pair evaluation CSV.",
    )
    args = parser.parse_args()

    predictions_path = args.predictions
    actual_path = args.actual
    if not predictions_path.exists():
        raise SystemExit(f"Predictions file {predictions_path} does not exist.")
    if not actual_path.exists():
        raise SystemExit(f"Actual metrics file {actual_path} does not exist.")

    predictions = load_predictions(predictions_path)
    actual, scale = load_actual_metrics(actual_path)
    if scale == 0.01:
        print("Detected percentage-form metrics; converted to fractions (0-1 scale).")

    evaluate_predictions(
        predictions,
        actual,
        base_threshold=args.base_threshold,
        donor_threshold=args.donor_threshold,
        per_pair_path=args.per_pair_output,
    )


if __name__ == "__main__":
    main()
