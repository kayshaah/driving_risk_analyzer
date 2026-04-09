"""
Result object produced by the risk analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskAnalysisResult:
    """Store detailed analysis output for one driving session"""

    heuristic_score: float
    heuristic_label: str
    model_label: str
    final_score: float
    final_label: str
    confidence: float
    feature_values: dict[str, float]
    recommendations: list[str]
