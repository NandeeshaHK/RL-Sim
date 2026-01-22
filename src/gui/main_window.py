"""Main application window for Rainbow Lunar Lander."""

import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QMenuBar, QMenu, QStatusBar,
    QMessageBox, QFileDialog, QApplication, QSplitter
)

from src.gui.panels.environment_panel import EnvironmentPanel
from src.gui.panels.network_panel import NetworkPanel
from src.gui.panels.metrics_panel import MetricsPanel
from src.gui.panels.control_panel import ControlPanel


class MainWindow(QMainWindow):
    """Main application window with dockable panels.
    
    Layout:
    - Left: Control panel
    - Center-Top: Environment + Network visualization
    - Center-Bottom: Metrics graphs
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the main window.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        
        self.config = config or {}
        
        self.setWindowTitle("Rainbow Lunar Lander - RL Experimentation Platform")
        self.setMinimumSize(1400, 900)
        
        # Load stylesheet
        self._load_stylesheet()
        
        # Setup UI
        self._setup_menu()
        self._setup_panels()
        self._setup_status_bar()
        
        # Connect signals
        self._connect_signals()
        
        # Training state
        self.is_training = False
        self.trainer = None

    def _load_stylesheet(self) -> None:
        """Load the dark theme stylesheet."""
        style_path = Path(__file__).parent / "styles" / "dark_theme.qss"
        
        if style_path.exists():
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        else:
            # Fallback minimal dark theme
            self.setStyleSheet("""
                QMainWindow { background-color: #0a0a1a; }
                QWidget { background-color: #0a0a1a; color: #e0e0e0; }
            """)

    def _setup_menu(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Session", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_session)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Checkpoint", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)
        
        load_action = QAction("&Load Checkpoint", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self._on_load)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        self.toggle_control_action = QAction("&Control Panel", self)
        self.toggle_control_action.setCheckable(True)
        self.toggle_control_action.setChecked(True)
        view_menu.addAction(self.toggle_control_action)
        
        self.toggle_network_action = QAction("&Network Panel", self)
        self.toggle_network_action.setCheckable(True)
        self.toggle_network_action.setChecked(True)
        view_menu.addAction(self.toggle_network_action)
        
        self.toggle_metrics_action = QAction("&Metrics Panel", self)
        self.toggle_metrics_action.setCheckable(True)
        self.toggle_metrics_action.setChecked(True)
        view_menu.addAction(self.toggle_metrics_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_panels(self) -> None:
        """Setup the main panels and layout."""
        # Central widget with splitters
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Horizontal splitter (control | main content)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Control panel (left sidebar)
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(280)
        h_splitter.addWidget(self.control_panel)
        
        # Main content area (vertical splitter)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top row: Environment + Network
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.env_panel = EnvironmentPanel()
        top_splitter.addWidget(self.env_panel)
        
        self.network_panel = NetworkPanel()
        top_splitter.addWidget(self.network_panel)
        
        top_splitter.setSizes([600, 400])  # More space to environment
        v_splitter.addWidget(top_splitter)
        
        # Bottom: Metrics
        self.metrics_panel = MetricsPanel()
        v_splitter.addWidget(self.metrics_panel)
        
        v_splitter.setSizes([550, 350])  # Better proportion for graphs
        h_splitter.addWidget(v_splitter)
        
        h_splitter.setSizes([250, 1150])  # Slightly narrower control panel
        main_layout.addWidget(h_splitter)
        
        # Connect view toggle actions
        self.toggle_control_action.triggered.connect(
            lambda checked: self.control_panel.setVisible(checked)
        )
        self.toggle_network_action.triggered.connect(
            lambda checked: self.network_panel.setVisible(checked)
        )
        self.toggle_metrics_action.triggered.connect(
            lambda checked: self.metrics_panel.setVisible(checked)
        )

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_bar.showMessage("Ready - Press Play to start training")

    def _connect_signals(self) -> None:
        """Connect panel signals to handlers."""
        # Control panel signals
        self.control_panel.play_clicked.connect(self._on_play)
        self.control_panel.pause_clicked.connect(self._on_pause)
        self.control_panel.stop_clicked.connect(self._on_stop)
        self.control_panel.reset_clicked.connect(self._on_reset)
        self.control_panel.speed_changed.connect(self._on_speed_changed)
        self.control_panel.algorithm_changed.connect(self._on_algorithm_changed)
        self.control_panel.save_clicked.connect(self._on_save)
        self.control_panel.load_clicked.connect(self._on_load)
        self.control_panel.add_instance_clicked.connect(self._on_add_instance)
        self.control_panel.remove_instance_clicked.connect(self._on_remove_instance)

    # === Event Handlers ===

    def _on_play(self) -> None:
        """Start or resume training."""
        self.is_training = True
        self.status_bar.showMessage("Training...")

    def _on_pause(self) -> None:
        """Pause training."""
        self.is_training = False
        self.status_bar.showMessage("Paused")

    def _on_stop(self) -> None:
        """Stop training."""
        self.is_training = False
        self.status_bar.showMessage("Stopped")

    def _on_reset(self) -> None:
        """Reset training session."""
        self.is_training = False
        self.env_panel.clear()
        self.metrics_panel.clear_all()
        self.status_bar.showMessage("Reset - Ready to start")

    def _on_speed_changed(self, speed: int) -> None:
        """Handle speed change."""
        if speed == 0:
            self.status_bar.showMessage("Speed: Maximum")
        else:
            self.status_bar.showMessage(f"Speed: {speed}x")

    def _on_algorithm_changed(self, algorithm: str) -> None:
        """Handle algorithm selection change."""
        self.status_bar.showMessage(f"Algorithm: {algorithm}")

    def _on_save(self) -> None:
        """Save checkpoint."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", "checkpoints/", "Checkpoint (*.pt)"
        )
        if path:
            self.status_bar.showMessage(f"Saved to {path}")

    def _on_load(self) -> None:
        """Load checkpoint."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", "checkpoints/", "Checkpoint (*.pt)"
        )
        if path:
            self.status_bar.showMessage(f"Loaded from {path}")

    def _on_add_instance(self) -> None:
        """Add new environment instance."""
        count = self.env_panel.tab_widget.count() + 1
        self.env_panel.add_instance(f"Instance {count}")
        self.status_bar.showMessage(f"Added Instance {count}")

    def _on_remove_instance(self) -> None:
        """Remove current environment instance."""
        current_idx = self.env_panel.tab_widget.currentIndex()
        if self.env_panel.tab_widget.count() > 1:
            self.env_panel.remove_instance(current_idx)
            self.status_bar.showMessage("Removed instance")

    def _on_new_session(self) -> None:
        """Start a new training session."""
        reply = QMessageBox.question(
            self,
            "New Session",
            "Start a new session? Current progress will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._on_reset()

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Rainbow Lunar Lander",
            "<h2>Rainbow Lunar Lander</h2>"
            "<p>Version 0.1.0</p>"
            "<p>A modular RL experimentation platform with real-time visualization.</p>"
            "<p>Algorithms: Double DQN, Rainbow DQN</p>"
            "<p>Environment: Lunar Lander v2</p>"
        )

    # === Public Methods ===

    def update_training_state(
        self,
        episode: int,
        timestep: int,
        total_timestep: int,
        epsilon: float,
        reward: float,
        q_values: Any = None,
        state: Any = None,
        frame: Any = None,
        v_value: float | None = None,
        loss: float | None = None,
    ) -> None:
        """Update the display with current training state.
        
        Args:
            episode: Current episode.
            timestep: Current episode timestep.
            total_timestep: Total timesteps.
            epsilon: Exploration rate.
            reward: Episode reward.
            q_values: Q-values for network display.
            state: Current environment state.
            frame: Environment render frame.
            v_value: State value estimate.
            loss: Training loss.
        """
        self.env_panel.update_stats(
            episode=episode,
            timestep=timestep,
            total_timestep=total_timestep,
            epsilon=epsilon,
            reward=reward,
        )
        
        if frame is not None:
            self.env_panel.update_frame(frame)
        
        if q_values is not None and state is not None:
            self.network_panel.update_values(
                input_values=state,
                q_values=q_values,
            )
        
        # Update step metrics for graphs
        if v_value is not None or loss is not None:
            self.metrics_panel.update_step_metrics(
                timestep=total_timestep,
                v_value=v_value,
                loss=loss,
            )
        
        self.control_panel.update_epsilon(epsilon)
