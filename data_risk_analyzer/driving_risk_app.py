"""
Tkinter GUI for the Intelligent Driving Behavior Risk Analyzer.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from driving_risk_analyzer.dataset_manager import DatasetManager
from driving_risk_analyzer.risk_analyzer import RiskAnalyzer


class DrivingRiskApp(tk.Tk):
    """Main graphical interface"""

    def __init__(
        self,
        default_sensor_csv: Path | None = None,
        default_reference_csv: Path | None = None,
    ) -> None:
        super().__init__()
        self.title("Intelligent Driving Behavior Risk Analyzer")
        self.geometry("1240x800")
        self.minsize(1040, 720)

        self.dataset_manager: DatasetManager | None = None
        self.analyzer = RiskAnalyzer(default_reference_csv)

        self.session_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Combined")
        self.sensitivity_var = tk.IntVar(value=3)
        self.sensitivity_label_var = tk.StringVar(value="3")
        self.status_var = tk.StringVar(
            value="Load a driving CSV file or use the bundled sample dataset."
        )
        self.metric_vars = {
            "risk_score": tk.StringVar(value="0.0"),
            "risk_label": tk.StringVar(value="Not analyzed"),
            "confidence": tk.StringVar(value="0.0%"),
            "avg_speed": tk.StringVar(value="0.0 km/h"),
            "max_speed": tk.StringVar(value="0.0 km/h"),
            "duration": tk.StringVar(value="0.0 s"),
        }

        self._build_layout()
        self._apply_style()

        if default_sensor_csv is not None:
            self.load_sensor_dataset(default_sensor_csv)

    def _build_layout(self) -> None:
        """Create the visual layout and controls."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=18)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Intelligent Driving Behavior Risk Analyzer",
            font=("Helvetica", 24, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=(
                "Upload motion-sensor driving data, choose a driving session, and"
                " analyze risky behavior using feature extraction and KNN-style classification."
            ),
            wraplength=1080,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        controls = ttk.LabelFrame(self, text="Controls", padding=14)
        controls.grid(row=1, column=0, sticky="ew", padx=18)
        for column in range(6):
            controls.columnconfigure(column, weight=1)

        ttk.Button(
            controls,
            text="Load Driving CSV",
            command=self._browse_sensor_csv,
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        ttk.Label(controls, text="Session").grid(row=0, column=1, sticky="w", padx=4)
        self.session_combo = ttk.Combobox(
            controls,
            textvariable=self.session_var,
            state="readonly",
            values=[],
        )
        self.session_combo.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(controls, text="Analysis Mode").grid(
            row=0,
            column=2,
            sticky="w",
            padx=4,
        )
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            state="readonly",
            values=["Combined", "Heuristic Only", "KNN Model"],
        )
        mode_combo.grid(row=1, column=2, sticky="ew", padx=4, pady=4)

        ttk.Label(controls, text="Sensitivity").grid(row=0, column=3, sticky="w", padx=4)
        ttk.Scale(
            controls,
            from_=1,
            to=5,
            variable=self.sensitivity_var,
            orient="horizontal",
            command=self._update_sensitivity_label,
        ).grid(row=1, column=3, sticky="ew", padx=4, pady=4)
        ttk.Label(controls, textvariable=self.sensitivity_label_var).grid(
            row=2,
            column=3,
            sticky="w",
            padx=4,
        )

        ttk.Button(
            controls,
            text="Analyze Session",
            command=self._analyze_selected_session,
        ).grid(row=1, column=4, sticky="ew", padx=4, pady=4)
        ttk.Button(
            controls,
            text="Reset View",
            command=self._reset_view,
        ).grid(row=1, column=5, sticky="ew", padx=4, pady=4)

        main_panel = ttk.Frame(self, padding=(18, 14, 18, 0))
        main_panel.grid(row=2, column=0, sticky="nsew")
        main_panel.columnconfigure(0, weight=3)
        main_panel.columnconfigure(1, weight=2)
        main_panel.rowconfigure(1, weight=1)

        metrics = ttk.LabelFrame(main_panel, text="Risk Summary", padding=12)
        metrics.grid(row=0, column=0, columnspan=2, sticky="ew")
        for column in range(6):
            metrics.columnconfigure(column, weight=1)

        self._create_metric_card(metrics, 0, "Risk Score", self.metric_vars["risk_score"])
        self._create_metric_card(metrics, 1, "Classification", self.metric_vars["risk_label"])
        self._create_metric_card(metrics, 2, "Confidence", self.metric_vars["confidence"])
        self._create_metric_card(metrics, 3, "Avg Speed", self.metric_vars["avg_speed"])
        self._create_metric_card(metrics, 4, "Max Speed", self.metric_vars["max_speed"])
        self._create_metric_card(metrics, 5, "Duration", self.metric_vars["duration"])

        feature_panel = ttk.LabelFrame(main_panel, text="Extracted Features", padding=12)
        feature_panel.grid(row=1, column=0, sticky="nsew", pady=(14, 0), padx=(0, 10))
        feature_panel.columnconfigure(0, weight=1)
        feature_panel.rowconfigure(0, weight=1)

        self.feature_table = ttk.Treeview(
            feature_panel,
            columns=("feature", "value"),
            show="headings",
            height=14,
        )
        self.feature_table.heading("feature", text="Feature")
        self.feature_table.heading("value", text="Value")
        self.feature_table.column("feature", width=270, anchor="w")
        self.feature_table.column("value", width=120, anchor="center")
        self.feature_table.grid(row=0, column=0, sticky="nsew")

        feature_scroll = ttk.Scrollbar(
            feature_panel,
            orient="vertical",
            command=self.feature_table.yview,
        )
        feature_scroll.grid(row=0, column=0, sticky="nse")
        self.feature_table.configure(yscrollcommand=feature_scroll.set)

        insight_panel = ttk.LabelFrame(main_panel, text="Interpretation", padding=12)
        insight_panel.grid(row=1, column=1, sticky="nsew", pady=(14, 0))
        insight_panel.columnconfigure(0, weight=1)
        insight_panel.rowconfigure(0, weight=1)

        self.insight_text = tk.Text(
            insight_panel,
            wrap="word",
            height=18,
            font=("Helvetica", 11),
        )
        self.insight_text.grid(row=0, column=0, sticky="nsew")
        insight_scroll = ttk.Scrollbar(
            insight_panel,
            orient="vertical",
            command=self.insight_text.yview,
        )
        insight_scroll.grid(row=0, column=1, sticky="ns")
        self.insight_text.configure(yscrollcommand=insight_scroll.set)

        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(10, 8),
        )
        status_bar.grid(row=3, column=0, sticky="ew", padx=18, pady=18)

    def _apply_style(self) -> None:
        """Apply minor readability tweaks."""
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=28)

    def _create_metric_card(
        self,
        parent: ttk.LabelFrame,
        column: int,
        title: str,
        variable: tk.StringVar,
    ) -> None:
        """Create a small summary card."""
        card = ttk.Frame(parent, padding=10)
        card.grid(row=0, column=column, sticky="ew")
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text=title, font=("Helvetica", 11, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(card, textvariable=variable, font=("Helvetica", 16)).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 0),
        )

    def _browse_sensor_csv(self) -> None:
        """Ask the user for a CSV file and load it."""
        selected_path = filedialog.askopenfilename(
            title="Select driving sensor CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected_path:
            self.load_sensor_dataset(Path(selected_path))

    def load_sensor_dataset(self, csv_path: Path) -> None:
        """Load sensor data into the app."""
        try:
            self.dataset_manager = DatasetManager.from_csv(csv_path)
        except Exception as error:
            messagebox.showerror("Unable to load sensor data", str(error))
            self.status_var.set("The selected sensor CSV could not be loaded.")
            return

        session_ids = self.dataset_manager.session_ids
        self.session_combo.configure(values=session_ids)
        if session_ids:
            self.session_var.set(session_ids[0])
            self._analyze_selected_session()
        self.status_var.set(f"Loaded sensor dataset: {csv_path.name}")

    def _update_sensitivity_label(self, raw_value: str) -> None:
        """Display the current sensitivity setting."""
        self.sensitivity_label_var.set(str(int(float(raw_value))))

    def _analyze_selected_session(self) -> None:
        """Run analysis for the currently selected session."""
        if self.dataset_manager is None:
            messagebox.showinfo("No data loaded", "Please load a driving CSV file first.")
            return

        session_id = self.session_var.get().strip()
        if not session_id:
            messagebox.showinfo("No session selected", "Choose a session to analyze.")
            return

        try:
            session = self.dataset_manager.build_session(session_id)
            session_frame = self.dataset_manager.get_session_frame(session_id)
            result = self.analyzer.analyze_session(
                session_frame=session_frame,
                analysis_mode=self.mode_var.get(),
                sensitivity=self.sensitivity_var.get(),
            )
        except Exception as error:
            messagebox.showerror("Analysis failed", str(error))
            self.status_var.set("Session analysis could not be completed.")
            return

        session.risk_score = result.final_score
        session.risk_label = result.final_label
        self._update_metrics(session, result)
        self._populate_feature_table(result.feature_values)
        self._populate_insights(result)
        self.status_var.set(
            f"Analyzed session {session.session_id} with mode {self.mode_var.get()}."
        )

    def _update_metrics(self, session, result) -> None:
        """Refresh top-line metrics for the selected session."""
        self.metric_vars["risk_score"].set(f"{result.final_score:.1f}")
        self.metric_vars["risk_label"].set(result.final_label.title())
        self.metric_vars["confidence"].set(f"{result.confidence:.1f}%")
        self.metric_vars["avg_speed"].set(f"{session.average_speed_kmh:.1f} km/h")
        self.metric_vars["max_speed"].set(f"{session.max_speed_kmh:.1f} km/h")
        self.metric_vars["duration"].set(f"{session.duration_seconds:.1f} s")

    def _populate_feature_table(self, feature_values: dict[str, float]) -> None:
        """Replace displayed feature rows."""
        for item_id in self.feature_table.get_children():
            self.feature_table.delete(item_id)

        for feature_name, display_value in self.analyzer.format_feature_rows(feature_values):
            self.feature_table.insert("", "end", values=(feature_name, display_value))

    def _populate_insights(self, result) -> None:
        """Write a readable explanation of the analysis."""
        self.insight_text.delete("1.0", tk.END)
        lines = [
            f"Heuristic label: {result.heuristic_label.title()}",
            f"KNN label: {result.model_label.title()}",
            f"Final label: {result.final_label.title()}",
            f"Risk score: {result.final_score:.1f}",
            f"Confidence: {result.confidence:.1f}%",
            "",
            "Recommendations:",
        ]
        for recommendation in result.recommendations:
            lines.append(f"- {recommendation}")

        self.insight_text.insert(tk.END, "\n".join(lines))

    def _reset_view(self) -> None:
        """Restore the default control state and clear previous output."""
        self.mode_var.set("Combined")
        self.sensitivity_var.set(3)
        self.sensitivity_label_var.set("3")

        for key, default_value in {
            "risk_score": "0.0",
            "risk_label": "Not analyzed",
            "confidence": "0.0%",
            "avg_speed": "0.0 km/h",
            "max_speed": "0.0 km/h",
            "duration": "0.0 s",
        }.items():
            self.metric_vars[key].set(default_value)

        for item_id in self.feature_table.get_children():
            self.feature_table.delete(item_id)
        self.insight_text.delete("1.0", tk.END)

        if self.dataset_manager and self.dataset_manager.session_ids:
            self.session_var.set(self.dataset_manager.session_ids[0])
        self.status_var.set("View reset. Choose a session and analyze it again.")
