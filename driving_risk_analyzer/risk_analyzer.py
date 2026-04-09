"""
Feature extraction, heuristic scoring, and KNN-style risk prediction.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd

from driving_risk_analyzer.analysis_result import RiskAnalysisResult


class SimpleRiskKNN:
    """A small custom KNN-style classifier built on reference feature profiles."""

    FEATURE_COLUMNS = [
        "average_speed_kmh",
        "max_speed_kmh",
        "speed_variability",
        "mean_abs_accel_x",
        "harsh_acceleration_events",
        "harsh_braking_events",
        "sharp_turn_events",
        "swerving_events",
        "overspeed_events",
    ]

    def __init__(self, reference_frame: pd.DataFrame) -> None:
        """Store validated training data."""
        self.reference_frame = self._prepare_reference_frame(reference_frame)
        self._bounds = self._compute_feature_bounds()

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "SimpleRiskKNN":
        """Load reference profiles from a CSV file."""
        frame = pd.read_csv(csv_path)
        return cls(frame)

    def predict(self, feature_values: dict[str, float], k: int = 3) -> tuple[str, float]:
        """Predict a label and confidence from a feature vector."""
        if self.reference_frame.empty:
            raise ValueError("Reference profiles are required for KNN prediction.")

        normalized_target = self._normalize_features(feature_values)
        distances: list[tuple[float, str]] = []
        for row in self.reference_frame.itertuples(index=False):
            row_features = {column: float(getattr(row, column)) for column in self.FEATURE_COLUMNS}
            normalized_row = self._normalize_features(row_features)
            squared_sum = 0.0
            for column in self.FEATURE_COLUMNS:
                squared_sum += (normalized_target[column] - normalized_row[column]) ** 2
            distances.append((sqrt(squared_sum), row.label))

        nearest = sorted(distances, key=lambda item: item[0])[: max(1, int(k))]
        label_weights: dict[str, float] = {}
        for distance, label in nearest:
            weight = 1 / max(distance, 0.001)
            label_weights[label] = label_weights.get(label, 0.0) + weight

        predicted_label = max(label_weights, key=label_weights.get)
        total_weight = sum(label_weights.values())
        confidence = round((label_weights[predicted_label] / total_weight) * 100, 1)
        return predicted_label, confidence

    def _prepare_reference_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the reference dataset."""
        required_columns = ["profile_id", *self.FEATURE_COLUMNS, "label"]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Reference CSV is missing required columns: {missing_text}")

        working = frame[required_columns].copy()
        if working.isnull().any().any():
            raise ValueError("Reference CSV contains blank values.")

        for column in self.FEATURE_COLUMNS:
            working[column] = pd.to_numeric(working[column], errors="raise")

        working["label"] = working["label"].astype(str).str.strip().str.lower()
        invalid_labels = ~working["label"].isin({"safe", "moderate", "high"})
        if invalid_labels.any():
            raise ValueError("Reference labels must be safe, moderate, or high.")

        return working.reset_index(drop=True)

    def _compute_feature_bounds(self) -> dict[str, tuple[float, float]]:
        """Capture min and max values used for feature normalization."""
        bounds: dict[str, tuple[float, float]] = {}
        for column in self.FEATURE_COLUMNS:
            series = self.reference_frame[column]
            bounds[column] = (float(series.min()), float(series.max()))
        return bounds

    def _normalize_features(self, feature_values: dict[str, float]) -> dict[str, float]:
        """Scale all features into roughly the same range."""
        normalized: dict[str, float] = {}
        for column in self.FEATURE_COLUMNS:
            lower, upper = self._bounds[column]
            value = float(feature_values[column])
            if upper == lower:
                normalized[column] = 0.0
            else:
                normalized[column] = (value - lower) / (upper - lower)
        return normalized


class RiskAnalyzer:
    """Compute driver risk features and produce risk predictions."""

    LABEL_TO_LEVEL = {"safe": 1, "moderate": 2, "high": 3}
    LEVEL_TO_LABEL = {1: "safe", 2: "moderate", 3: "high"}
    FEATURE_LABELS = {
        "average_speed_kmh": "Average Speed (km/h)",
        "max_speed_kmh": "Max Speed (km/h)",
        "speed_variability": "Speed Variability",
        "mean_abs_accel_x": "Mean |Longitudinal Acceleration|",
        "harsh_acceleration_events": "Harsh Acceleration Events",
        "harsh_braking_events": "Harsh Braking Events",
        "sharp_turn_events": "Sharp Turn Events",
        "swerving_events": "Swerving Events",
        "overspeed_events": "Overspeed Events",
        "duration_seconds": "Duration (seconds)",
    }

    def __init__(self, reference_csv_path: str | Path | None = None) -> None:
        """Load the reference classifier when a CSV path is provided."""
        self.classifier = (
            SimpleRiskKNN.from_csv(reference_csv_path)
            if reference_csv_path is not None
            else None
        )

    def extract_features(self, session_frame: pd.DataFrame) -> dict[str, float]:
        """Convert raw sensor rows into feature values for risk analysis."""
        working = session_frame.sort_values("timestamp_sec").reset_index(drop=True)

        accel_x = working["accel_x"]
        accel_y = working["accel_y"]
        gyro_z = working["gyro_z"]
        speed = working["speed_kmh"]

        duration_seconds = float(
            max(0.0, working["timestamp_sec"].max() - working["timestamp_sec"].min())
        )

        features = {
            "average_speed_kmh": round(float(speed.mean()), 2),
            "max_speed_kmh": round(float(speed.max()), 2),
            "speed_variability": round(float(speed.std(ddof=0)), 2),
            "mean_abs_accel_x": round(float(accel_x.abs().mean()), 2),
            "harsh_acceleration_events": float(
                self._count_event_bursts(accel_x >= 2.8)
            ),
            "harsh_braking_events": float(self._count_event_bursts(accel_x <= -3.0)),
            "sharp_turn_events": float(self._count_event_bursts(gyro_z.abs() >= 35)),
            "swerving_events": float(self._count_event_bursts(accel_y.abs() >= 2.5)),
            "overspeed_events": float(self._count_event_bursts(speed >= 90)),
            "duration_seconds": round(duration_seconds, 2),
        }
        return features

    def analyze_session(
        self,
        session_frame: pd.DataFrame,
        analysis_mode: str = "Combined",
        sensitivity: int = 3,
    ) -> RiskAnalysisResult:
        """Analyze one session using heuristic, KNN, or combined logic."""
        features = self.extract_features(session_frame)

        heuristic_score = self._score_with_rules(features, sensitivity=sensitivity)
        heuristic_label = self._label_from_score(heuristic_score)

        model_label = heuristic_label
        model_confidence = 0.0
        if self.classifier is not None:
            model_label, model_confidence = self.classifier.predict(features)

        final_score, final_label, confidence = self._combine_predictions(
            heuristic_score=heuristic_score,
            heuristic_label=heuristic_label,
            model_label=model_label,
            model_confidence=model_confidence,
            analysis_mode=analysis_mode,
        )

        return RiskAnalysisResult(
            heuristic_score=heuristic_score,
            heuristic_label=heuristic_label,
            model_label=model_label,
            final_score=final_score,
            final_label=final_label,
            confidence=confidence,
            feature_values=features,
            recommendations=self._build_recommendations(features, final_label),
        )

    def format_feature_rows(self, feature_values: dict[str, float]) -> list[tuple[str, str]]:
        """Prepare feature rows for GUI display."""
        order = [
            "duration_seconds",
            "average_speed_kmh",
            "max_speed_kmh",
            "speed_variability",
            "mean_abs_accel_x",
            "harsh_acceleration_events",
            "harsh_braking_events",
            "sharp_turn_events",
            "swerving_events",
            "overspeed_events",
        ]
        rows: list[tuple[str, str]] = []
        for key in order:
            label = self.FEATURE_LABELS.get(key, key)
            value = feature_values[key]
            if float(value).is_integer():
                rows.append((label, str(int(value))))
            else:
                rows.append((label, f"{value:.2f}"))
        return rows

    def _score_with_rules(self, feature_values: dict[str, float], sensitivity: int) -> float:
        """Generate a risk score using hand-built logic."""
        base_score = (
            6 * feature_values["harsh_acceleration_events"]
            + 8 * feature_values["harsh_braking_events"]
            + 5 * feature_values["sharp_turn_events"]
            + 5 * feature_values["swerving_events"]
            + 4 * feature_values["overspeed_events"]
            + 0.6 * feature_values["speed_variability"]
            + 5 * feature_values["mean_abs_accel_x"]
        )
        sensitivity = min(5, max(1, int(sensitivity)))
        sensitivity_multiplier = 1 + ((sensitivity - 3) * 0.08)
        return round(min(100.0, base_score * sensitivity_multiplier), 1)

    def _label_from_score(self, score: float) -> str:
        """Convert a numeric score into a risk label."""
        if score < 35:
            return "safe"
        if score < 65:
            return "moderate"
        return "high"

    def _combine_predictions(
        self,
        heuristic_score: float,
        heuristic_label: str,
        model_label: str,
        model_confidence: float,
        analysis_mode: str,
    ) -> tuple[float, str, float]:
        """Merge heuristic and classifier output into a final view."""
        model_score = {
            "safe": 20.0,
            "moderate": 55.0,
            "high": 85.0,
        }[model_label]

        heuristic_confidence = self._heuristic_confidence(heuristic_score)
        normalized_mode = analysis_mode.strip().casefold()

        if normalized_mode == "heuristic only":
            return heuristic_score, heuristic_label, heuristic_confidence

        if normalized_mode == "knn model":
            return model_score, model_label, max(model_confidence, 50.0)

        final_score = round((heuristic_score * 0.7) + (model_score * 0.3), 1)
        final_label = self._label_from_score(final_score)
        agreement_bonus = 10.0 if heuristic_label == model_label else 0.0
        confidence = round(
            min(
                99.0,
                (heuristic_confidence * 0.45) + (model_confidence * 0.45) + agreement_bonus,
            ),
            1,
        )
        return final_score, final_label, confidence

    def _heuristic_confidence(self, score: float) -> float:
        """Estimate confidence from distance to label boundaries."""
        boundaries = [35.0, 65.0]
        distance = min(abs(score - boundary) for boundary in boundaries)
        return round(min(95.0, 55.0 + distance), 1)

    def _build_recommendations(
        self,
        feature_values: dict[str, float],
        final_label: str,
    ) -> list[str]:
        """Generate suggestions based on the extracted features."""
        recommendations: list[str] = []

        if feature_values["harsh_braking_events"] > 0:
            recommendations.append("Reduce abrupt braking by leaving more following distance.")
        if feature_values["harsh_acceleration_events"] > 0:
            recommendations.append("Smooth out rapid acceleration to improve control and safety.")
        if feature_values["sharp_turn_events"] > 0 or feature_values["swerving_events"] > 0:
            recommendations.append("Make steering inputs more gradual, especially during turns.")
        if feature_values["overspeed_events"] > 0:
            recommendations.append("Reduce high-speed driving and keep speed more consistent.")

        if not recommendations and final_label == "safe":
            recommendations.append("Driving behavior appears steady. Continue maintaining smooth inputs.")

        if feature_values["speed_variability"] > 18:
            recommendations.append("Large speed swings were detected. Try maintaining a steadier pace.")

        return recommendations
