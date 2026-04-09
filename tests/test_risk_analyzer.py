"""Pytest Tests for driving data loading and risk analysis."""

from _future_ import annotations

from pathlib import Path

import pandas as pd
import pytest

from driving_risk_analyzer.dataset_manager import DatasetManager
from driving_risk_analyzer.risk_analyzer import RiskAnalyzer, SimpleRiskKNN


@pytest.fixture
def data_paths() -> dict[str, Path]:
    """Provide paths to bundled sample data files."""
    project_root = Path(_file_).resolve().parents[1]
    return {
        "sensor_csv": project_root / "data" / "sample_driving_sensor_data.csv",
        "reference_csv": project_root / "data" / "risk_reference_profiles.csv",
    }


@pytest.fixture
def dataset_manager(data_paths: dict[str, Path]) -> DatasetManager:
    """Load the bundled sensor dataset."""
    return DatasetManager.from_csv(data_paths["sensor_csv"])


@pytest.fixture
def risk_analyzer(data_paths: dict[str, Path]) -> RiskAnalyzer:
    """Create an analyzer backed by bundled reference profiles."""
    return RiskAnalyzer(data_paths["reference_csv"])


def test_dataset_manager_loads_all_sessions(dataset_manager: DatasetManager) -> None:
    """The dataset should expose every distinct session."""
    assert dataset_manager.session_ids == ["101", "202", "303"]
    session = dataset_manager.build_session("101")
    assert session.row_count == 12
    assert session.duration_seconds == 55.0
    assert session.average_speed_kmh == 37.0


def test_extract_features_for_moderate_session(
    dataset_manager: DatasetManager,
    risk_analyzer: RiskAnalyzer,
) -> None:
    """Feature extraction should count driving events from the sensor data."""
    session_frame = dataset_manager.get_session_frame("202")
    features = risk_analyzer.extract_features(session_frame)
    assert features["harsh_acceleration_events"] == 2.0
    assert features["harsh_braking_events"] == 1.0
    assert features["sharp_turn_events"] == 2.0
    assert features["swerving_events"] == 2.0
    assert features["overspeed_events"] == 0.0


def test_safe_session_scores_as_safe(
    dataset_manager: DatasetManager,
    risk_analyzer: RiskAnalyzer,
) -> None:
    """The low-risk sample session should be classified as safe."""
    result = risk_analyzer.analyze_session(
        dataset_manager.get_session_frame("101"),
        analysis_mode="Combined",
        sensitivity=3,
    )
    assert result.final_label == "safe"
    assert result.final_score < 35


def test_moderate_session_scores_as_moderate(
    dataset_manager: DatasetManager,
    risk_analyzer: RiskAnalyzer,
) -> None:
    """The middle sample session should land in the moderate range."""
    result = risk_analyzer.analyze_session(
        dataset_manager.get_session_frame("202"),
        analysis_mode="Combined",
        sensitivity=3,
    )
    assert result.final_label == "moderate"
    assert 35 <= result.final_score < 65


def test_high_session_scores_as_high(
    dataset_manager: DatasetManager,
    risk_analyzer: RiskAnalyzer,
) -> None:
    """The aggressive sample session should be classified as high risk."""
    result = risk_analyzer.analyze_session(
        dataset_manager.get_session_frame("303"),
        analysis_mode="Combined",
        sensitivity=3,
    )
    assert result.final_label == "high"
    assert result.final_score >= 65


def test_knn_predicts_expected_label_for_high_risk_profile(
    data_paths: dict[str, Path],
) -> None:
    """The reference classifier should recognize a high-risk feature vector."""
    classifier = SimpleRiskKNN.from_csv(data_paths["reference_csv"])
    label, confidence = classifier.predict(
        {
            "average_speed_kmh": 79.0,
            "max_speed_kmh": 108.0,
            "speed_variability": 19.0,
            "mean_abs_accel_x": 3.2,
            "harsh_acceleration_events": 3.0,
            "harsh_braking_events": 2.0,
            "sharp_turn_events": 3.0,
            "swerving_events": 3.0,
            "overspeed_events": 2.0,
        }
    )
    assert label == "high"
    assert confidence > 50


def test_missing_required_column_raises_error() -> None:
    """Invalid sensor data should fail validation."""
    bad_frame = pd.DataFrame(
        {
            "session_id": ["1"],
            "timestamp_sec": [0],
            "accel_x": [0.5],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        DatasetManager(bad_frame)
