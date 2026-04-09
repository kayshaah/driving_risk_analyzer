"""Main.py file to launch the Driving Behavior Risk Analyzer GUI."""

from pathlib import Path

from driving_risk_analyzer.driving_risk_app import DrivingRiskApp


def main() -> None:
    """Start the GUI"""
    project_root = Path(__file__).resolve().parent
    default_sensor_csv = project_root / "data" / "sample_driving_sensor_data.csv"
    default_reference_csv = project_root / "data" / "risk_reference_profiles.csv"

    app = DrivingRiskApp(
        default_sensor_csv=default_sensor_csv if default_sensor_csv.exists() else None,
        default_reference_csv=default_reference_csv if default_reference_csv.exists() else None,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
