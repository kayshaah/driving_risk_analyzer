"""Tkinter GUI for the Intelligent Driving Behavior Risk Analyzer."""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from driving_risk_analyzer.dataset_manager import DatasetManager
from driving_risk_analyzer.report_exporter import write_html_report
from driving_risk_analyzer.risk_analyzer import RiskAnalyzer


class DrivingRiskApp(tk.Tk):
    """Main graphical interface for the driving-risk project."""

    COLORS = {
        "navy": "#102a43",
        "blue": "#1d4ed8",
        "blue_light": "#dbeafe",
        "red": "#dc2626",
        "red_light": "#fee2e2",
        "white": "#ffffff",
        "surface": "#f4f7fb",
        "line": "#cbd5e1",
        "text": "#0f172a",
        "muted": "#64748b",
        "safe": "#2563eb",
        "moderate": "#475569",
        "high": "#dc2626",
    }

    def __init__(
        self,
        default_sensor_csv: Path | None = None,
        default_reference_csv: Path | None = None,
    ) -> None:
        super().__init__()
        self.title("Intelligent Driving Behavior Risk Analyzer")
        self.geometry("1260x840")
        self.minsize(1060, 740)
        self.configure(bg=self.COLORS["surface"])

        self.project_root = Path(__file__).resolve().parents[1]
        self.logo_path = self.project_root / "assets" / "driveguard_logo.png"
        self.sample_csv_path = self.project_root / "data" / "sample_driving_sensor_data.csv"
        self.extended_csv_path = self.project_root / "data" / "extended_driving_sensor_data.csv"

        self.dataset_manager: DatasetManager | None = None
        self.analyzer = RiskAnalyzer(default_reference_csv)

        self.session_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Combined")
        self.sensitivity_var = tk.IntVar(value=3)
        self.sensitivity_label_var = tk.StringVar(value="3")
        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(
            value="Load a driving CSV file or use one of the sample datasets."
        )
        self.metric_vars = {
            "risk_score": tk.StringVar(value="0.0"),
            "risk_label": tk.StringVar(value="Not analyzed"),
            "confidence": tk.StringVar(value="0.0%"),
            "avg_speed": tk.StringVar(value="0.0 km/h"),
            "max_speed": tk.StringVar(value="0.0 km/h"),
            "duration": tk.StringVar(value="0.0 s"),
        }
        self.metric_value_labels: dict[str, tk.Label] = {}
        self.current_feature_values: dict[str, float] = {}
        self.last_session = None
        self.last_result = None
        self.logo_image = self._load_logo_image()

        self._apply_style()
        self._build_layout()

        if default_sensor_csv is not None:
            self.load_sensor_dataset(default_sensor_csv)

    def _build_layout(self) -> None:
        """Create the visual layout and controls."""
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.scroll_canvas = tk.Canvas(
            self,
            bg=self.COLORS["surface"],
            highlightthickness=0,
        )
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")

        page_scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.scroll_canvas.yview,
        )
        page_scrollbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=page_scrollbar.set)

        self.scrollable_page = tk.Frame(self.scroll_canvas, bg=self.COLORS["surface"])
        self.scroll_window = self.scroll_canvas.create_window(
            (0, 0),
            window=self.scrollable_page,
            anchor="nw",
        )
        self.scrollable_page.columnconfigure(0, weight=1)
        self.scrollable_page.bind("<Configure>", self._update_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._resize_scroll_window)
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_header(self.scrollable_page)
        self._build_controls(self.scrollable_page)
        self._build_dashboard(self.scrollable_page)
        self._build_status_bar()

    def _build_header(self, parent: tk.Frame) -> None:
        """Create a bold dashboard header."""
        header = tk.Frame(parent, bg=self.COLORS["navy"], padx=22, pady=14)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title = tk.Label(
            header,
            text="Intelligent Driving Behavior Risk Analyzer",
            bg=self.COLORS["navy"],
            fg=self.COLORS["white"],
            font=("Helvetica", 25, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = tk.Label(
            header,
            text=(
                "Analyze motion-sensor trips, compare rule-based and KNN results, "
                "and spot risky driving events at a glance."
            ),
            bg=self.COLORS["navy"],
            fg="#d7e3f4",
            font=("Helvetica", 11),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(6, 0))

        if self.logo_image is not None:
            tk.Label(
                header,
                image=self.logo_image,
                bg=self.COLORS["navy"],
            ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(24, 0))
        else:
            tk.Label(
                header,
                text="DRIVEGUARD",
                bg=self.COLORS["navy"],
                fg=self.COLORS["white"],
                font=("Helvetica", 13, "bold"),
            ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(24, 0))

    def _build_controls(self, parent: tk.Frame) -> None:
        """Create the interactive control strip."""
        controls = tk.Frame(
            parent,
            bg=self.COLORS["white"],
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
            padx=14,
            pady=14,
        )
        controls.grid(row=1, column=0, sticky="ew", padx=18, pady=(16, 0))
        for column in range(10):
            controls.columnconfigure(column, weight=1)

        ttk.Button(
            controls,
            text="Load CSV/ZIP",
            style="Accent.TButton",
            command=self._browse_sensor_csv,
        ).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(
            controls,
            text="Sample",
            style="Secondary.TButton",
            command=self._load_sample_dataset,
        ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            controls,
            text="Extended",
            style="Secondary.TButton",
            command=self._load_extended_dataset,
        ).grid(row=0, column=2, sticky="ew", padx=4)

        ttk.Label(controls, text="Session", style="Control.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=4,
            pady=(12, 2),
        )
        self.session_combo = ttk.Combobox(
            controls,
            textvariable=self.session_var,
            state="readonly",
            values=[],
        )
        self.session_combo.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4)
        self.session_combo.bind("<<ComboboxSelected>>", lambda _event: self._auto_analyze())

        ttk.Label(controls, text="Analysis Mode", style="Control.TLabel").grid(
            row=1,
            column=2,
            sticky="w",
            padx=4,
            pady=(12, 2),
        )
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            state="readonly",
            values=["Combined", "Heuristic Only", "KNN Model"],
        )
        mode_combo.grid(row=2, column=2, columnspan=2, sticky="ew", padx=4)
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._auto_analyze())

        ttk.Label(controls, text="Risk Sensitivity", style="Control.TLabel").grid(
            row=1,
            column=4,
            sticky="w",
            padx=4,
            pady=(12, 2),
        )
        ttk.Scale(
            controls,
            from_=1,
            to=5,
            variable=self.sensitivity_var,
            orient="horizontal",
            command=self._update_sensitivity_label,
        ).grid(row=2, column=4, sticky="ew", padx=4)
        tk.Label(
            controls,
            textvariable=self.sensitivity_label_var,
            bg=self.COLORS["blue_light"],
            fg=self.COLORS["blue"],
            font=("Helvetica", 11, "bold"),
            width=3,
        ).grid(row=2, column=5, sticky="w", padx=4)

        ttk.Checkbutton(
            controls,
            text="Auto refresh",
            variable=self.auto_refresh_var,
            style="Control.TCheckbutton",
        ).grid(row=2, column=6, sticky="w", padx=4)

        ttk.Button(
            controls,
            text="Analyze Session",
            style="Primary.TButton",
            command=self._analyze_selected_session,
        ).grid(row=2, column=7, sticky="ew", padx=4)
        ttk.Button(
            controls,
            text="Export Report",
            style="Secondary.TButton",
            command=self._export_report,
        ).grid(row=2, column=8, sticky="ew", padx=4)
        ttk.Button(
            controls,
            text="Reset",
            style="Secondary.TButton",
            command=self._reset_view,
        ).grid(row=2, column=9, sticky="ew", padx=4)

    def _build_dashboard(self, parent: tk.Frame) -> None:
        """Build the result dashboard."""
        dashboard = tk.Frame(parent, bg=self.COLORS["surface"], padx=18, pady=16)
        dashboard.grid(row=2, column=0, sticky="nsew")
        dashboard.columnconfigure(0, weight=2)
        dashboard.columnconfigure(1, weight=3)
        dashboard.rowconfigure(0, weight=1)

        left_column = tk.Frame(dashboard, bg=self.COLORS["surface"])
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_column.columnconfigure(0, weight=1)

        right_column = tk.Frame(dashboard, bg=self.COLORS["surface"])
        right_column.grid(row=0, column=1, sticky="nsew")
        right_column.columnconfigure(0, weight=1)
        right_column.rowconfigure(0, weight=2)
        right_column.rowconfigure(1, weight=1)

        self._build_risk_meter(left_column)
        self._build_metric_cards(left_column)
        self._build_event_bars(left_column)
        self._build_feature_table(right_column)
        self._build_insights(right_column)

    def _build_risk_meter(self, parent: tk.Frame) -> None:
        """Create a canvas-based risk gauge."""
        panel = self._create_panel(parent, "Risk Meter")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)

        self.risk_canvas = tk.Canvas(
            panel,
            width=380,
            height=240,
            bg=self.COLORS["white"],
            highlightthickness=0,
        )
        self.risk_canvas.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._draw_risk_meter(0.0, "not analyzed")

    def _build_metric_cards(self, parent: tk.Frame) -> None:
        """Create colorful top-line metrics."""
        panel = self._create_panel(parent, "Trip Summary")
        panel.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for column in range(2):
            panel.columnconfigure(column, weight=1)

        cards = [
            ("risk_score", "Risk Score", self.COLORS["red_light"], self.COLORS["red"]),
            ("risk_label", "Classification", self.COLORS["blue_light"], self.COLORS["blue"]),
            ("confidence", "Confidence", "#eef2ff", self.COLORS["navy"]),
            ("avg_speed", "Avg Speed", "#e0f2fe", "#0369a1"),
            ("max_speed", "Max Speed", "#fef2f2", self.COLORS["red"]),
            ("duration", "Duration", "#f8fafc", self.COLORS["text"]),
        ]
        for index, (key, title, background, accent) in enumerate(cards):
            card = tk.Frame(
                panel,
                bg=background,
                highlightbackground="#e2e8f0",
                highlightthickness=1,
                padx=10,
                pady=8,
            )
            card.grid(row=(index // 2) + 1, column=index % 2, sticky="ew", padx=5, pady=5)
            card.columnconfigure(0, weight=1)
            tk.Label(
                card,
                text=title,
                bg=background,
                fg=self.COLORS["muted"],
                font=("Helvetica", 9, "bold"),
            ).grid(row=0, column=0, sticky="w")
            value_label = tk.Label(
                card,
                textvariable=self.metric_vars[key],
                bg=background,
                fg=accent,
                font=("Helvetica", 15, "bold"),
            )
            value_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.metric_value_labels[key] = value_label

    def _build_event_bars(self, parent: tk.Frame) -> None:
        """Create a small graphic showing risky event counts."""
        panel = self._create_panel(parent, "Event Breakdown")
        panel.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        panel.columnconfigure(0, weight=1)

        self.event_canvas = tk.Canvas(
            panel,
            width=380,
            height=190,
            bg=self.COLORS["white"],
            highlightthickness=0,
        )
        self.event_canvas.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._draw_event_bars({})

    def _build_feature_table(self, parent: tk.Frame) -> None:
        """Create the extracted feature table."""
        panel = self._create_panel(parent, "Extracted Features")
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        self.feature_table = ttk.Treeview(
            panel,
            columns=("feature", "value"),
            show="headings",
            height=13,
            style="Feature.Treeview",
        )
        self.feature_table.heading("feature", text="Feature")
        self.feature_table.heading("value", text="Value")
        self.feature_table.column("feature", width=340, anchor="w")
        self.feature_table.column("value", width=140, anchor="center")
        self.feature_table.grid(row=1, column=0, sticky="nsew")

        feature_scroll = ttk.Scrollbar(
            panel,
            orient="vertical",
            command=self.feature_table.yview,
        )
        feature_scroll.grid(row=1, column=1, sticky="ns")
        self.feature_table.configure(yscrollcommand=feature_scroll.set)

    def _build_insights(self, parent: tk.Frame) -> None:
        """Create the textual interpretation panel."""
        panel = self._create_panel(parent, "Interpretation")
        panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        self.insight_text = tk.Text(
            panel,
            wrap="word",
            height=9,
            font=("Helvetica", 11),
            bg=self.COLORS["white"],
            fg=self.COLORS["text"],
            padx=12,
            pady=10,
            bd=0,
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
        )
        self.insight_text.grid(row=1, column=0, sticky="nsew")
        self.insight_text.tag_configure("heading", foreground=self.COLORS["navy"], font=("Helvetica", 12, "bold"))
        self.insight_text.tag_configure("safe", foreground=self.COLORS["safe"], font=("Helvetica", 11, "bold"))
        self.insight_text.tag_configure("moderate", foreground=self.COLORS["moderate"], font=("Helvetica", 11, "bold"))
        self.insight_text.tag_configure("high", foreground=self.COLORS["high"], font=("Helvetica", 11, "bold"))

        insight_scroll = ttk.Scrollbar(
            panel,
            orient="vertical",
            command=self.insight_text.yview,
        )
        insight_scroll.grid(row=1, column=1, sticky="ns")
        self.insight_text.configure(yscrollcommand=insight_scroll.set)

    def _build_status_bar(self) -> None:
        """Create the bottom status bar."""
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            bg=self.COLORS["navy"],
            fg=self.COLORS["white"],
            anchor="w",
            padx=14,
            pady=9,
            font=("Helvetica", 10),
        )
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _update_scroll_region(self, _event: tk.Event) -> None:
        """Keep the page scrollbar matched to the content height."""
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _resize_scroll_window(self, event: tk.Event) -> None:
        """Let the scrollable content fill the visible window width."""
        self.scroll_canvas.itemconfigure(self.scroll_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Scroll the page with the mouse wheel or trackpad."""
        if not hasattr(self, "scroll_canvas"):
            return
        direction = -1 if event.delta > 0 else 1
        self.scroll_canvas.yview_scroll(direction, "units")

    def _apply_style(self) -> None:
        """Apply the red, white, and blue visual theme."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=self.COLORS["surface"])
        style.configure(
            "Control.TLabel",
            background=self.COLORS["white"],
            foreground=self.COLORS["muted"],
            font=("Helvetica", 9, "bold"),
        )
        style.configure("TCombobox", padding=4)
        style.configure(
            "Control.TCheckbutton",
            background=self.COLORS["white"],
            foreground=self.COLORS["text"],
            font=("Helvetica", 10),
        )
        style.configure(
            "Primary.TButton",
            background=self.COLORS["blue"],
            foreground=self.COLORS["white"],
            borderwidth=0,
            padding=(12, 8),
            font=("Helvetica", 10, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#1e40af")])
        style.configure(
            "Accent.TButton",
            background=self.COLORS["red"],
            foreground=self.COLORS["white"],
            borderwidth=0,
            padding=(12, 8),
            font=("Helvetica", 10, "bold"),
        )
        style.map("Accent.TButton", background=[("active", "#b91c1c")])
        style.configure(
            "Secondary.TButton",
            background="#eef2ff",
            foreground=self.COLORS["navy"],
            borderwidth=0,
            padding=(12, 8),
            font=("Helvetica", 10, "bold"),
        )
        style.map("Secondary.TButton", background=[("active", "#dbeafe")])
        style.configure(
            "Feature.Treeview",
            rowheight=29,
            background=self.COLORS["white"],
            fieldbackground=self.COLORS["white"],
            foreground=self.COLORS["text"],
            borderwidth=0,
            font=("Helvetica", 10),
        )
        style.configure(
            "Feature.Treeview.Heading",
            background=self.COLORS["blue"],
            foreground=self.COLORS["white"],
            font=("Helvetica", 10, "bold"),
        )
        style.map("Feature.Treeview", background=[("selected", self.COLORS["blue"])])

    def _create_panel(self, parent: tk.Frame, title: str) -> tk.Frame:
        """Create a reusable white dashboard panel."""
        panel = tk.Frame(
            parent,
            bg=self.COLORS["white"],
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        tk.Label(
            panel,
            text=title,
            bg=self.COLORS["white"],
            fg=self.COLORS["navy"],
            font=("Helvetica", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        return panel

    def _load_logo_image(self) -> tk.PhotoImage | None:
        """Load the DriveGuard logo and use it as the app/window icon."""
        if not self.logo_path.exists():
            return None

        try:
            logo_image = tk.PhotoImage(file=str(self.logo_path))
            self.iconphoto(True, logo_image)
            return logo_image
        except tk.TclError:
            return None

    def _draw_risk_meter(self, score: float, label: str) -> None:
        """Draw a semicircle risk gauge."""
        canvas = self.risk_canvas
        canvas.delete("all")
        width = int(canvas.cget("width"))
        center_x = width // 2
        center_y = 168
        radius = 125
        box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)

        canvas.create_arc(box, start=180, extent=-63, style="arc", width=20, outline=self.COLORS["blue"])
        canvas.create_arc(box, start=117, extent=-54, style="arc", width=20, outline="#94a3b8")
        canvas.create_arc(box, start=63, extent=-63, style="arc", width=20, outline=self.COLORS["red"])

        clamped_score = max(0.0, min(100.0, float(score)))
        angle = 180 - (clamped_score * 1.8)
        needle_length = 92
        end_x = center_x + needle_length * cos(radians(angle))
        end_y = center_y - needle_length * sin(radians(angle))
        color = self._risk_color(label)

        canvas.create_line(center_x, center_y, end_x, end_y, fill=color, width=5)
        canvas.create_oval(center_x - 8, center_y - 8, center_x + 8, center_y + 8, fill=color, outline="")
        canvas.create_text(
            center_x,
            74,
            text=f"{clamped_score:.1f}",
            fill=color,
            font=("Helvetica", 34, "bold"),
        )
        canvas.create_text(
            center_x,
            108,
            text=label.title(),
            fill=self.COLORS["text"],
            font=("Helvetica", 14, "bold"),
        )
        canvas.create_text(60, 202, text="SAFE", fill=self.COLORS["blue"], font=("Helvetica", 9, "bold"))
        canvas.create_text(center_x, 216, text="MODERATE", fill=self.COLORS["moderate"], font=("Helvetica", 9, "bold"))
        canvas.create_text(width - 58, 202, text="HIGH", fill=self.COLORS["red"], font=("Helvetica", 9, "bold"))

    def _draw_event_bars(self, feature_values: dict[str, float]) -> None:
        """Draw horizontal bars for event counts."""
        canvas = self.event_canvas
        canvas.delete("all")

        events = [
            ("Accel", feature_values.get("harsh_acceleration_events", 0.0), self.COLORS["blue"]),
            ("Brake", feature_values.get("harsh_braking_events", 0.0), self.COLORS["red"]),
            ("Turns", feature_values.get("sharp_turn_events", 0.0), "#334155"),
            ("Swerve", feature_values.get("swerving_events", 0.0), "#64748b"),
            ("Speed", feature_values.get("overspeed_events", 0.0), self.COLORS["red"]),
        ]
        max_value = max([value for _label, value, _color in events] + [1.0])

        for index, (label, value, color) in enumerate(events):
            y = 24 + index * 31
            canvas.create_text(18, y, text=label, anchor="w", fill=self.COLORS["text"], font=("Helvetica", 10, "bold"))
            canvas.create_rectangle(88, y - 8, 315, y + 8, fill="#eef2f7", outline="")
            bar_width = 0 if max_value == 0 else (value / max_value) * 227
            canvas.create_rectangle(88, y - 8, 88 + bar_width, y + 8, fill=color, outline="")
            canvas.create_text(344, y, text=str(int(value)), fill=self.COLORS["text"], font=("Helvetica", 10, "bold"))

    def _browse_sensor_csv(self) -> None:
        """Ask the user for a CSV file and load it."""
        selected_path = filedialog.askopenfilename(
            title="Select driving sensor CSV or Sensor Logger ZIP",
            filetypes=[
                ("Driving data", "*.csv *.zip"),
                ("CSV files", "*.csv"),
                ("ZIP files", "*.zip"),
                ("All files", "*.*"),
            ],
        )
        if selected_path:
            self.load_sensor_dataset(Path(selected_path))

    def _load_sample_dataset(self) -> None:
        """Load the bundled three-session sample dataset."""
        self._load_known_dataset(self.sample_csv_path)

    def _load_extended_dataset(self) -> None:
        """Load the larger bundled test dataset."""
        self._load_known_dataset(self.extended_csv_path)

    def _load_known_dataset(self, csv_path: Path) -> None:
        """Load a bundled dataset if it exists."""
        if not csv_path.exists():
            messagebox.showerror("Dataset not found", f"Could not find {csv_path.name}.")
            return
        self.load_sensor_dataset(csv_path)

    def load_sensor_dataset(self, csv_path: Path) -> None:
        """Load sensor data into the app."""
        try:
            if csv_path.suffix.casefold() == ".zip":
                self.dataset_manager = DatasetManager.from_zip(csv_path)
            else:
                self.dataset_manager = DatasetManager.from_csv(csv_path)
        except Exception as error:
            messagebox.showerror("Unable to load sensor data", str(error))
            self.status_var.set("The selected CSV or ZIP file could not be loaded.")
            return

        session_ids = self.dataset_manager.session_ids
        self.session_combo.configure(values=session_ids)
        if session_ids:
            self.session_var.set(session_ids[0])
            self._analyze_selected_session()
        source_type = "ZIP sensor export" if csv_path.suffix.casefold() == ".zip" else "sensor dataset"
        self.status_var.set(f"Loaded {source_type}: {csv_path.name}")

    def _update_sensitivity_label(self, raw_value: str) -> None:
        """Display the current sensitivity setting."""
        self.sensitivity_label_var.set(str(int(float(raw_value))))
        self._auto_analyze()

    def _auto_analyze(self) -> None:
        """Analyze automatically when the user changes important controls."""
        if self.auto_refresh_var.get() and self.dataset_manager is not None:
            self._analyze_selected_session(show_messages=False)

    def _analyze_selected_session(self, show_messages: bool = True) -> None:
        """Run analysis for the currently selected session."""
        if self.dataset_manager is None:
            if show_messages:
                messagebox.showinfo("No data loaded", "Please load a driving CSV file first.")
            return

        session_id = self.session_var.get().strip()
        if not session_id:
            if show_messages:
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
            if show_messages:
                messagebox.showerror("Analysis failed", str(error))
            self.status_var.set("Session analysis could not be completed.")
            return

        session.risk_score = result.final_score
        session.risk_label = result.final_label
        self.current_feature_values = result.feature_values
        self.last_session = session
        self.last_result = result
        self._update_metrics(session, result)
        self._populate_feature_table(result.feature_values)
        self._populate_insights(result)
        self._draw_risk_meter(result.final_score, result.final_label)
        self._draw_event_bars(result.feature_values)
        self.status_var.set(
            f"Analyzed session {session.session_id} with mode {self.mode_var.get()}."
        )

    def _export_report(self) -> None:
        """Export the latest analysis as a branded HTML report."""
        if self.last_session is None or self.last_result is None:
            self._analyze_selected_session(show_messages=True)

        if self.last_session is None or self.last_result is None:
            return

        default_name = self._suggest_report_filename(self.last_session.session_id)
        selected_path = filedialog.asksaveasfilename(
            title="Export DriveGuard report",
            defaultextension=".html",
            initialfile=default_name,
            filetypes=[
                ("HTML report", "*.html"),
                ("All files", "*.*"),
            ],
        )
        if not selected_path:
            return

        try:
            report_path = write_html_report(
                output_path=selected_path,
                session=self.last_session,
                result=self.last_result,
                logo_path=self.logo_path,
            )
        except Exception as error:
            messagebox.showerror("Export failed", str(error))
            self.status_var.set("The driving report could not be exported.")
            return

        self.status_var.set(f"Exported report: {report_path.name}")
        messagebox.showinfo("Report exported", f"Saved DriveGuard report to:\n{report_path}")

    def _update_metrics(self, session, result) -> None:
        """Refresh top-line metrics for the selected session."""
        self.metric_vars["risk_score"].set(f"{result.final_score:.1f}")
        self.metric_vars["risk_label"].set(result.final_label.title())
        self.metric_vars["confidence"].set(f"{result.confidence:.1f}%")
        self.metric_vars["avg_speed"].set(f"{session.average_speed_kmh:.1f} km/h")
        self.metric_vars["max_speed"].set(f"{session.max_speed_kmh:.1f} km/h")
        self.metric_vars["duration"].set(f"{session.duration_seconds:.1f} s")

        label_color = self._risk_color(result.final_label)
        self.metric_value_labels["risk_score"].configure(fg=label_color)
        self.metric_value_labels["risk_label"].configure(fg=label_color)

    def _populate_feature_table(self, feature_values: dict[str, float]) -> None:
        """Replace displayed feature rows."""
        for item_id in self.feature_table.get_children():
            self.feature_table.delete(item_id)

        for index, (feature_name, display_value) in enumerate(self.analyzer.format_feature_rows(feature_values)):
            tag = "even" if index % 2 == 0 else "odd"
            self.feature_table.insert("", "end", values=(feature_name, display_value), tags=(tag,))
        self.feature_table.tag_configure("even", background=self.COLORS["white"])
        self.feature_table.tag_configure("odd", background="#f8fafc")

    def _populate_insights(self, result) -> None:
        """Write a readable explanation of the analysis."""
        self.insight_text.delete("1.0", tk.END)

        features = result.feature_values
        total_events = int(
            features["harsh_acceleration_events"]
            + features["harsh_braking_events"]
            + features["sharp_turn_events"]
            + features["swerving_events"]
            + features["overspeed_events"]
        )
        dominant_event = self._dominant_event_summary(features)

        self.insight_text.insert(tk.END, "Driving Review\n", "heading")
        self.insight_text.insert(
            tk.END,
            self._build_overall_review(result.final_label, result.final_score, result.confidence),
        )
        self.insight_text.insert(tk.END, "\n\n")

        self.insight_text.insert(tk.END, "Model Comparison\n", "heading")
        self.insight_text.insert(tk.END, "Heuristic label: ")
        self.insight_text.insert(tk.END, f"{result.heuristic_label.title()}\n", result.heuristic_label)
        self.insight_text.insert(tk.END, "KNN label: ")
        self.insight_text.insert(tk.END, f"{result.model_label.title()}\n", result.model_label)
        self.insight_text.insert(tk.END, "Final label: ")
        self.insight_text.insert(tk.END, f"{result.final_label.title()}\n", result.final_label)
        self.insight_text.insert(tk.END, f"Risk score: {result.final_score:.1f}\n")
        self.insight_text.insert(tk.END, f"Confidence: {result.confidence:.1f}%\n\n")

        self.insight_text.insert(tk.END, "Behavior Details\n", "heading")
        self.insight_text.insert(
            tk.END,
            (
                f"- Total risky event bursts detected: {total_events}\n"
                f"- Most noticeable behavior pattern: {dominant_event}\n"
                f"- Average speed: {features['average_speed_kmh']:.1f} km/h\n"
                f"- Maximum speed: {features['max_speed_kmh']:.1f} km/h\n"
                f"- Speed variability: {features['speed_variability']:.1f}\n"
            ),
        )
        self.insight_text.insert(tk.END, self._build_event_review(features))
        self.insight_text.insert(tk.END, "\n")

        self.insight_text.insert(tk.END, "Recommendations\n", "heading")
        for recommendation in result.recommendations:
            self.insight_text.insert(tk.END, f"- {recommendation}\n")
        for recommendation in self._build_extra_recommendations(features, result.final_label):
            self.insight_text.insert(tk.END, f"- {recommendation}\n")

    def _build_overall_review(self, label: str, score: float, confidence: float) -> str:
        """Create a plain-language summary of the session result."""
        normalized_label = label.strip().casefold()
        if normalized_label == "safe":
            message = (
                "This session shows mostly controlled driving. The score suggests "
                "smooth inputs with few risky motion patterns."
            )
        elif normalized_label == "moderate":
            message = (
                "This session shows some warning signs. The driver was not consistently "
                "unsafe, but there were enough motion changes to deserve attention."
            )
        else:
            message = (
                "This session shows a high-risk pattern. The score indicates repeated "
                "aggressive movement or unstable driving behavior."
            )

        return f"{message} Final score: {score:.1f}/100. Confidence: {confidence:.1f}%."

    def _dominant_event_summary(self, features: dict[str, float]) -> str:
        """Summarize the largest event category in user-facing language."""
        event_labels = {
            "harsh_acceleration_events": "rapid acceleration",
            "harsh_braking_events": "abrupt braking",
            "sharp_turn_events": "sharp turning",
            "swerving_events": "side-to-side movement",
            "overspeed_events": "overspeeding",
        }
        event_counts = {
            key: int(features.get(key, 0.0))
            for key in event_labels
        }
        largest_key = max(event_counts, key=event_counts.get)
        largest_value = event_counts[largest_key]
        if largest_value == 0:
            return "no major risky event category stood out"
        return f"{event_labels[largest_key]} ({largest_value} bursts)"

    def _build_event_review(self, features: dict[str, float]) -> str:
        """Create a detailed event-by-event interpretation."""
        lines: list[str] = []
        if features["harsh_acceleration_events"] > 0:
            lines.append(
                f"- Rapid acceleration appeared {int(features['harsh_acceleration_events'])} times, which can reduce reaction time."
            )
        if features["harsh_braking_events"] > 0:
            lines.append(
                f"- Abrupt braking appeared {int(features['harsh_braking_events'])} times, often linked to short following distance or late reactions."
            )
        if features["sharp_turn_events"] > 0:
            lines.append(
                f"- Sharp turning appeared {int(features['sharp_turn_events'])} times, suggesting stronger steering inputs."
            )
        if features["swerving_events"] > 0:
            lines.append(
                f"- Swerving appeared {int(features['swerving_events'])} times, suggesting lateral instability."
            )
        if features["overspeed_events"] > 0:
            lines.append(
                f"- Overspeeding appeared {int(features['overspeed_events'])} times based on the configured threshold."
            )
        if not lines:
            lines.append("- No harsh braking, rapid acceleration, sharp turn, swerving, or overspeed bursts were detected.")
        return "\n".join(lines) + "\n"

    def _build_extra_recommendations(
        self,
        features: dict[str, float],
        label: str,
    ) -> list[str]:
        """Add extra feedback based on combined feature patterns."""
        recommendations: list[str] = []
        total_events = int(
            features["harsh_acceleration_events"]
            + features["harsh_braking_events"]
            + features["sharp_turn_events"]
            + features["swerving_events"]
            + features["overspeed_events"]
        )

        if total_events >= 8:
            recommendations.append("Review the trip segment carefully because several risky events occurred close together.")
        if features["harsh_acceleration_events"] > 0 and features["harsh_braking_events"] > 0:
            recommendations.append("Avoid repeated speed-up and slow-down patterns by keeping a smoother following distance.")
        if features["max_speed_kmh"] > 80 and features["speed_variability"] > 15:
            recommendations.append("High speed combined with speed changes can increase risk; keep speed steady when traffic allows.")
        if label == "moderate":
            recommendations.append("This trip is a good candidate for coaching because small improvements could move it into the safe range.")
        if label == "high":
            recommendations.append("For a high-risk trip, focus first on braking distance, smoother steering, and avoiding sudden acceleration.")
        if not recommendations:
            recommendations.append("Keep collecting more sessions so the project can compare this drive against a broader driver history.")
        return recommendations

    def _reset_view(self) -> None:
        """Restore the default control state and clear previous output."""
        self.mode_var.set("Combined")
        self.sensitivity_var.set(3)
        self.sensitivity_label_var.set("3")
        self.current_feature_values = {}
        self.last_session = None
        self.last_result = None

        for key, default_value in {
            "risk_score": "0.0",
            "risk_label": "Not analyzed",
            "confidence": "0.0%",
            "avg_speed": "0.0 km/h",
            "max_speed": "0.0 km/h",
            "duration": "0.0 s",
        }.items():
            self.metric_vars[key].set(default_value)

        for key in ["risk_score", "risk_label"]:
            if key in self.metric_value_labels:
                self.metric_value_labels[key].configure(fg=self.COLORS["red"])

        for item_id in self.feature_table.get_children():
            self.feature_table.delete(item_id)
        self.insight_text.delete("1.0", tk.END)
        self._draw_risk_meter(0.0, "not analyzed")
        self._draw_event_bars({})

        if self.dataset_manager and self.dataset_manager.session_ids:
            self.session_var.set(self.dataset_manager.session_ids[0])
        self.status_var.set("View reset. Choose a session and analyze it again.")

    def _suggest_report_filename(self, session_id: str) -> str:
        """Build a file-system friendly default report name."""
        safe_session_id = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in session_id
        )
        return f"driveguard_report_{safe_session_id}.html"

    def _risk_color(self, label: str) -> str:
        """Return the UI color for a risk label."""
        normalized_label = label.strip().casefold()
        if normalized_label == "safe":
            return self.COLORS["safe"]
        if normalized_label == "moderate":
            return self.COLORS["moderate"]
        if normalized_label == "high":
            return self.COLORS["high"]
        return self.COLORS["red"]
