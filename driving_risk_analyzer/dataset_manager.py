"""
CSV loading and session management for driving sensor data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driving_risk_analyzer.driving_session import DrivingSession


class DatasetManager:
    """Load, validate, and split sensor data into driving sessions."""

    REQUIRED_COLUMNS = [
        "session_id",
        "timestamp_sec",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "speed_kmh",
    ]

    def __init__(self, frame: pd.DataFrame, source_name: str = "") -> None:
        """Create a manager from a DataFrame that already exists in memory."""
        self.source_name = source_name
        self._frame = self._prepare_frame(frame)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "DatasetManager":
        """Load sensor data from a CSV file."""
        csv_path = Path(csv_path)
        frame = pd.read_csv(csv_path)
        return cls(frame, source_name=csv_path.name)

    @property
    def frame(self) -> pd.DataFrame:
        """Return a copy of the validated underlying data."""
        return self._frame.copy()

    @property
    def session_ids(self) -> list[str]:
        """Return all available session identifiers in sorted order."""
        return sorted(self._frame["session_id"].unique().tolist())

    def get_session_frame(self, session_id: str) -> pd.DataFrame:
        """Return one validated session sorted by time."""
        filtered = self._frame[self._frame["session_id"] == str(session_id)].copy()
        if filtered.empty:
            raise ValueError(f"Session {session_id} was not found in the loaded dataset.")
        return filtered.sort_values("timestamp_sec").reset_index(drop=True)

    def build_session(self, session_id: str) -> DrivingSession:
        """Build a high-level DrivingSession summary from raw rows."""
        frame = self.get_session_frame(session_id)
        return DrivingSession(
            session_id=str(session_id),
            start_time_sec=float(frame["timestamp_sec"].min()),
            end_time_sec=float(frame["timestamp_sec"].max()),
            row_count=int(len(frame)),
            average_speed_kmh=round(float(frame["speed_kmh"].mean()), 1),
            max_speed_kmh=round(float(frame["speed_kmh"].max()), 1),
            source_name=self.source_name,
        )

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate columns and normalize types for analysis"""
        working = frame.copy()
        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in working.columns
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Sensor CSV is missing required columns: {missing_text}")

        working = working[self.REQUIRED_COLUMNS].copy()

        if working.isnull().any().any():
            raise ValueError("Sensor CSV contains blank values in required fields.")

        working["session_id"] = working["session_id"].astype(str).str.strip()
        if (working["session_id"] == "").any():
            raise ValueError("Every row must contain a non-empty session_id.")

        numeric_columns = [
            "timestamp_sec",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "speed_kmh",
        ]
        for column in numeric_columns:
            working[column] = pd.to_numeric(working[column], errors="raise")

        if (working["timestamp_sec"] < 0).any():
            raise ValueError("timestamp_sec values must be zero or greater.")

        if (working["speed_kmh"] < 0).any():
            raise ValueError("speed_kmh values must be zero or greater.")

        working = working.sort_values(["session_id", "timestamp_sec"]).reset_index(drop=True)
        return working
