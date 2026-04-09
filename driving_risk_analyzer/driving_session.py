"""
Data model for a summarized driving session.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DrivingSession:
    """Store high level information about one driving session."""

    session_id: str
    start_time_sec: float
    end_time_sec: float
    row_count: int
    average_speed_kmh: float
    max_speed_kmh: float
    source_name: str = ""
    risk_score: float | None = None
    risk_label: str = "Not analyzed"

    @property
    def duration_seconds(self) -> float:
        """Return the session duration in seconds."""
        return round(max(0.0, self.end_time_sec - self.start_time_sec), 1)

    @property
    def duration_minutes(self) -> float:
        """Return the session duration in minutes."""
        return round(self.duration_seconds / 60, 2)
