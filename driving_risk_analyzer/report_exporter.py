"""HTML report export for completed driving-risk analyses."""

from __future__ import annotations

from base64 import b64encode
from datetime import datetime
from html import escape
from pathlib import Path

from driving_risk_analyzer.analysis_result import RiskAnalysisResult
from driving_risk_analyzer.driving_session import DrivingSession


FEATURE_LABELS = {
    "duration_seconds": "Duration (seconds)",
    "average_speed_kmh": "Average Speed (km/h)",
    "max_speed_kmh": "Max Speed (km/h)",
    "speed_variability": "Speed Variability",
    "mean_abs_accel_x": "Mean |Longitudinal Acceleration|",
    "harsh_acceleration_events": "Harsh Acceleration Events",
    "harsh_braking_events": "Harsh Braking Events",
    "sharp_turn_events": "Sharp Turn Events",
    "swerving_events": "Swerving Events",
    "overspeed_events": "Overspeed Events",
}


def write_html_report(
    output_path: str | Path,
    session: DrivingSession,
    result: RiskAnalysisResult,
    logo_path: str | Path | None = None,
) -> Path:
    """Write a polished standalone HTML report and return its path."""
    output_path = Path(output_path)
    html = build_html_report(session=session, result=result, logo_path=logo_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_html_report(
    session: DrivingSession,
    result: RiskAnalysisResult,
    logo_path: str | Path | None = None,
) -> str:
    """Build a complete branded HTML report for one analyzed session."""
    logo_markup = _logo_markup(logo_path)
    risk_color = _risk_color(result.final_label)
    features = result.feature_values
    event_rows = _event_rows(features)
    feature_rows = _feature_rows(features)
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total_events = int(sum(value for _name, value, _color in event_rows))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DriveGuard Driving Risk Report</title>
  <style>
    :root {{
      --navy: #102a43;
      --blue: #1d4ed8;
      --red: #dc2626;
      --surface: #f4f7fb;
      --line: #cbd5e1;
      --text: #0f172a;
      --muted: #64748b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--surface);
      color: var(--text);
      font-family: Helvetica, Arial, sans-serif;
      line-height: 1.45;
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      background: var(--navy);
      color: white;
      padding: 24px 28px;
      border-radius: 8px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .hero p {{
      margin: 0;
      color: #d7e3f4;
    }}
    .logo {{
      width: 88px;
      height: 88px;
      object-fit: contain;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .card, .panel {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card span {{
      color: var(--muted);
      display: block;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
      text-transform: uppercase;
    }}
    .card strong {{
      color: var(--navy);
      display: block;
      font-size: 22px;
    }}
    .score strong {{
      color: {risk_color};
      font-size: 32px;
    }}
    h2 {{
      color: var(--navy);
      font-size: 18px;
      margin: 0 0 12px;
    }}
    .two-column {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    .bar-row {{
      align-items: center;
      display: grid;
      grid-template-columns: 135px 1fr 38px;
      gap: 10px;
      margin: 12px 0;
    }}
    .bar-track {{
      background: #eef2f7;
      border-radius: 999px;
      height: 14px;
      overflow: hidden;
    }}
    .bar-fill {{
      border-radius: 999px;
      height: 100%;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th {{
      background: var(--blue);
      color: white;
      text-align: left;
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 10px;
    }}
    tr:nth-child(even) td {{
      background: #f8fafc;
    }}
    .recommendations li {{
      margin: 8px 0;
    }}
    .footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 18px;
      text-align: center;
    }}
    @media (max-width: 850px) {{
      .grid, .two-column {{ grid-template-columns: 1fr; }}
      .hero {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <h1>DriveGuard Driving Risk Report</h1>
        <p>Session {escape(session.session_id)} exported on {escape(generated_at)}</p>
      </div>
      {logo_markup}
    </section>

    <section class="grid">
      <div class="card score"><span>Risk Score</span><strong>{result.final_score:.1f}</strong></div>
      <div class="card"><span>Classification</span><strong style="color:{risk_color}">{escape(result.final_label.title())}</strong></div>
      <div class="card"><span>Confidence</span><strong>{result.confidence:.1f}%</strong></div>
      <div class="card"><span>Total Risk Events</span><strong>{total_events}</strong></div>
      <div class="card"><span>Average Speed</span><strong>{session.average_speed_kmh:.1f} km/h</strong></div>
      <div class="card"><span>Max Speed</span><strong>{session.max_speed_kmh:.1f} km/h</strong></div>
      <div class="card"><span>Duration</span><strong>{session.duration_seconds:.1f} s</strong></div>
      <div class="card"><span>Data Source</span><strong>{escape(session.source_name or "Loaded dataset")}</strong></div>
    </section>

    <section class="two-column">
      <div class="panel">
        <h2>Model Summary</h2>
        <p>{escape(_overall_review(result.final_label, result.final_score, result.confidence))}</p>
        <p><strong>Heuristic label:</strong> {escape(result.heuristic_label.title())}</p>
        <p><strong>KNN label:</strong> {escape(result.model_label.title())}</p>
        <p><strong>Final label:</strong> {escape(result.final_label.title())}</p>
      </div>
      <div class="panel">
        <h2>Event Breakdown</h2>
        {_event_bar_markup(event_rows)}
      </div>
    </section>

    <section class="panel" style="margin-top:18px">
      <h2>Extracted Features</h2>
      <table>
        <thead><tr><th>Feature</th><th>Value</th></tr></thead>
        <tbody>
          {feature_rows}
        </tbody>
      </table>
    </section>

    <section class="panel recommendations" style="margin-top:18px">
      <h2>Recommendations</h2>
      <ul>
        {_recommendation_markup(result.recommendations, features, result.final_label)}
      </ul>
    </section>

    <p class="footer">Generated by DriveGuard Intelligent Driving Behavior Risk Analyzer.</p>
  </main>
</body>
</html>
"""


def _logo_markup(logo_path: str | Path | None) -> str:
    """Return an embedded logo image when available."""
    if logo_path is None:
        return '<div class="logo" aria-hidden="true"></div>'

    path = Path(logo_path)
    if not path.exists():
        return '<div class="logo" aria-hidden="true"></div>'

    encoded_logo = b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="logo" src="data:image/png;base64,{encoded_logo}" alt="DriveGuard logo">'


def _risk_color(label: str) -> str:
    """Return the report color for a risk label."""
    normalized_label = label.strip().casefold()
    if normalized_label == "safe":
        return "#2563eb"
    if normalized_label == "moderate":
        return "#475569"
    return "#dc2626"


def _event_rows(features: dict[str, float]) -> list[tuple[str, float, str]]:
    """Return event rows used for report bars."""
    return [
        ("Rapid acceleration", features.get("harsh_acceleration_events", 0.0), "#1d4ed8"),
        ("Abrupt braking", features.get("harsh_braking_events", 0.0), "#dc2626"),
        ("Sharp turns", features.get("sharp_turn_events", 0.0), "#334155"),
        ("Swerving", features.get("swerving_events", 0.0), "#64748b"),
        ("Overspeeding", features.get("overspeed_events", 0.0), "#dc2626"),
    ]


def _event_bar_markup(event_rows: list[tuple[str, float, str]]) -> str:
    """Build the event bar HTML."""
    max_count = max([count for _label, count, _color in event_rows] + [1.0])
    rows = []
    for label, count, color in event_rows:
        width = 0 if max_count == 0 else (count / max_count) * 100
        rows.append(
            f"""
            <div class="bar-row">
              <strong>{escape(label)}</strong>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div></div>
              <span>{int(count)}</span>
            </div>
            """
        )
    return "\n".join(rows)


def _feature_rows(features: dict[str, float]) -> str:
    """Build feature table rows."""
    rows = []
    for key, label in FEATURE_LABELS.items():
        value = features.get(key, 0.0)
        display_value = str(int(value)) if float(value).is_integer() else f"{value:.2f}"
        rows.append(f"<tr><td>{escape(label)}</td><td>{escape(display_value)}</td></tr>")
    return "\n".join(rows)


def _recommendation_markup(
    recommendations: list[str],
    features: dict[str, float],
    final_label: str,
) -> str:
    """Build recommendation list items."""
    combined = list(recommendations)
    total_events = sum(
        features.get(key, 0.0)
        for key in [
            "harsh_acceleration_events",
            "harsh_braking_events",
            "sharp_turn_events",
            "swerving_events",
            "overspeed_events",
        ]
    )

    if total_events >= 8:
        combined.append("Review this trip carefully because multiple risky events were detected.")
    if features.get("harsh_acceleration_events", 0.0) > 0 and features.get("harsh_braking_events", 0.0) > 0:
        combined.append("Work on smoother speed control to reduce repeated acceleration and braking cycles.")
    if final_label == "high":
        combined.append("Prioritize safer following distance, smoother steering, and more gradual acceleration.")
    if not combined:
        combined.append("Continue collecting driving sessions to build a stronger long-term safety profile.")

    return "\n".join(f"<li>{escape(item)}</li>" for item in combined)


def _overall_review(label: str, score: float, confidence: float) -> str:
    """Create a plain-language report summary."""
    normalized_label = label.strip().casefold()
    if normalized_label == "safe":
        message = "This session shows mostly controlled driving with few risky motion patterns."
    elif normalized_label == "moderate":
        message = "This session shows some warning signs that may benefit from coaching."
    else:
        message = "This session shows a high-risk driving pattern with repeated aggressive or unstable movement."
    return f"{message} Final score: {score:.1f}/100. Confidence: {confidence:.1f}%."
