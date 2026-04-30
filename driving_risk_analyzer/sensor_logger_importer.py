"""Import Sensor Logger"""

from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd


class SensorLoggerImportError(ValueError):
    """Raised when a Sensor Logger ZIP cannot be converted."""


def load_sensor_logger_zip(zip_path: str | Path) -> pd.DataFrame:
    """Load a Sensor Logger ZIP and return the normalized app DataFrame."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise SensorLoggerImportError(f"ZIP file was not found: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path) as archive:
            accelerometer = _load_motion_stream(
                archive,
                filename="Accelerometer.csv",
                axis_prefix="accel",
            )
            gyroscope = _load_motion_stream(
                archive,
                filename="Gyroscope.csv",
                axis_prefix="gyro",
            )
            location = _load_location_stream(archive)
    except zipfile.BadZipFile as error:
        raise SensorLoggerImportError("The selected file is not a valid ZIP file.") from error

    start_time = min(
        float(accelerometer["timestamp_sec"].min()),
        float(gyroscope["timestamp_sec"].min()),
    )
    accelerometer["timestamp_sec"] = accelerometer["timestamp_sec"] - start_time
    gyroscope["timestamp_sec"] = gyroscope["timestamp_sec"] - start_time
    location["timestamp_sec"] = location["timestamp_sec"] - start_time

    merged = pd.merge_asof(
        accelerometer.sort_values("timestamp_sec"),
        gyroscope.sort_values("timestamp_sec"),
        on="timestamp_sec",
        direction="nearest",
        tolerance=0.05,
    )
    merged = pd.merge_asof(
        merged.sort_values("timestamp_sec"),
        location[["timestamp_sec", "speed_kmh"]].sort_values("timestamp_sec"),
        on="timestamp_sec",
        direction="nearest",
        tolerance=3.0,
    )

    merged["speed_kmh"] = (
        merged["speed_kmh"].interpolate(limit_direction="both").bfill().ffill().fillna(0.0)
    )
    merged = merged.dropna(
        subset=["gyro_x", "gyro_y", "gyro_z", "speed_kmh"],
    ).copy()

    if merged.empty:
        raise SensorLoggerImportError(
            "The ZIP streams could not be aligned into a usable driving session."
        )

    merged.insert(0, "session_id", zip_path.stem)
    return merged[
        [
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
    ].reset_index(drop=True)


def _load_motion_stream(
    archive: zipfile.ZipFile,
    filename: str,
    axis_prefix: str,
) -> pd.DataFrame:
    """Load accelerometer or gyroscope rows from the ZIP."""
    frame = _read_csv_from_archive(archive, filename)
    required_columns = ["seconds_elapsed", "x", "y", "z"]
    _require_columns(frame, required_columns, filename)

    working = frame[required_columns].rename(
        columns={
            "seconds_elapsed": "timestamp_sec",
            "x": f"{axis_prefix}_x",
            "y": f"{axis_prefix}_y",
            "z": f"{axis_prefix}_z",
        }
    )
    return _coerce_numeric_frame(working, filename).dropna().sort_values(
        "timestamp_sec",
    ).reset_index(drop=True)


def _load_location_stream(archive: zipfile.ZipFile) -> pd.DataFrame:
    """Load GPS speed rows from Sensor Logger's Location.csv."""
    frame = _read_csv_from_archive(archive, "Location.csv")
    required_columns = ["seconds_elapsed", "speed"]
    _require_columns(frame, required_columns, "Location.csv")

    working = frame[required_columns].rename(
        columns={
            "seconds_elapsed": "timestamp_sec",
            "speed": "speed_mps",
        }
    )
    working = _coerce_numeric_frame(working, "Location.csv")
    working = working.dropna(subset=["timestamp_sec"]).sort_values(
        "timestamp_sec",
    ).reset_index(drop=True)
    if working.empty:
        raise SensorLoggerImportError("Location.csv does not contain usable GPS rows.")

    working.loc[working["speed_mps"] < 0, "speed_mps"] = pd.NA
    speed_mps = (
        working["speed_mps"].interpolate(limit_direction="both").bfill().ffill().fillna(0.0)
    )
    working["speed_kmh"] = (speed_mps * 3.6).round(4)
    return working[["timestamp_sec", "speed_kmh"]]


def _read_csv_from_archive(archive: zipfile.ZipFile, expected_filename: str) -> pd.DataFrame:
    """Read a CSV by basename so ZIPs may include folder prefixes."""
    matching_names = [
        name for name in archive.namelist() if Path(name).name == expected_filename
    ]
    if not matching_names:
        raise SensorLoggerImportError(f"The ZIP is missing {expected_filename}.")

    with archive.open(matching_names[0]) as csv_file:
        return pd.read_csv(csv_file)


def _require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    filename: str,
) -> None:
    """Raise a clear message when a Sensor Logger CSV is missing columns."""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise SensorLoggerImportError(f"{filename} is missing columns: {missing_text}")


def _coerce_numeric_frame(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Convert all fields in a sensor stream to numeric values."""
    working = frame.copy()
    for column in working.columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working["timestamp_sec"].isna().all():
        raise SensorLoggerImportError(f"{filename} does not contain usable timestamps.")
    return working
