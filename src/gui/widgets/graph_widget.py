"""Custom graph widgets for real-time plotting using PyQtGraph."""

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QCheckBox, QComboBox, QGroupBox
)


class GraphWidget(QWidget):
    """Single real-time graph with moving average support."""

    def __init__(
        self,
        title: str = "Graph",
        x_label: str = "X",
        y_label: str = "Y",
        max_points: int = 10000,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the graph widget.
        
        Args:
            title: Graph title.
            x_label: X-axis label.
            y_label: Y-axis label.
            max_points: Maximum data points to store.
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.title = title
        self.max_points = max_points
        
        # Data storage
        self.x_data: list[float] = []
        self.y_data: list[float] = []
        self.ma_windows: list[int] = [10, 50, 100]
        self.show_ma: dict[int, bool] = {w: False for w in self.ma_windows}
        
        self._setup_ui(x_label, y_label)

    def _setup_ui(self, x_label: str, y_label: str) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Configure PyQtGraph
        pg.setConfigOptions(antialias=True)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#0a0a1a")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setTitle(self.title, color="#ffd700", size="12pt")
        self.plot_widget.setLabel("left", y_label, color="#e0e0e0")
        self.plot_widget.setLabel("bottom", x_label, color="#e0e0e0")
        
        # Main data line
        self.main_line = self.plot_widget.plot(
            pen=pg.mkPen(color="#ff6b6b", width=1.5),
            name="Raw"
        )
        
        # Moving average lines
        self.ma_lines: dict[int, pg.PlotDataItem] = {}
        ma_colors = ["#4ecdc4", "#45b7d1", "#96ceb4"]
        
        for i, window in enumerate(self.ma_windows):
            color = ma_colors[i % len(ma_colors)]
            self.ma_lines[window] = self.plot_widget.plot(
                pen=pg.mkPen(color=color, width=2),
                name=f"MA{window}"
            )
            self.ma_lines[window].hide()
        
        # Add legend
        self.plot_widget.addLegend(offset=(10, 10))
        
        layout.addWidget(self.plot_widget)

    def add_point(self, x: float, y: float) -> None:
        """Add a new data point.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        self.x_data.append(x)
        self.y_data.append(y)
        
        # Trim if exceeding max points
        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]
        
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the plot with current data."""
        x = np.array(self.x_data)
        y = np.array(self.y_data)
        
        self.main_line.setData(x, y)
        
        # Update moving averages
        for window, line in self.ma_lines.items():
            if self.show_ma.get(window, False) and len(y) >= window:
                ma = self._compute_moving_average(y, window)
                line.setData(x[window-1:], ma)
                line.show()
            else:
                line.hide()

    @staticmethod
    def _compute_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average.
        
        Args:
            data: Input data array.
            window: Window size.
            
        Returns:
            Moving average array.
        """
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def set_moving_average(self, window: int, enabled: bool) -> None:
        """Enable/disable a moving average line.
        
        Args:
            window: Window size.
            enabled: Whether to show the MA line.
        """
        if window in self.show_ma:
            self.show_ma[window] = enabled
            self._update_plot()

    def set_ma_windows(self, windows: Sequence[int]) -> None:
        """Update the moving average windows.
        
        Args:
            windows: List of window sizes.
        """
        # Remove old lines
        for line in self.ma_lines.values():
            self.plot_widget.removeItem(line)
        
        self.ma_windows = list(windows)
        self.show_ma = {w: False for w in self.ma_windows}
        self.ma_lines = {}
        
        # Create new lines
        ma_colors = ["#4ecdc4", "#45b7d1", "#96ceb4", "#dfe6e9"]
        for i, window in enumerate(self.ma_windows):
            color = ma_colors[i % len(ma_colors)]
            self.ma_lines[window] = self.plot_widget.plot(
                pen=pg.mkPen(color=color, width=2),
                name=f"MA{window}"
            )
            self.ma_lines[window].hide()

    def clear(self) -> None:
        """Clear all data."""
        self.x_data.clear()
        self.y_data.clear()
        self.main_line.setData([], [])
        for line in self.ma_lines.values():
            line.setData([], [])


class MultiGraphWidget(QWidget):
    """Widget containing multiple synchronized graphs with MA controls."""

    ma_changed = pyqtSignal(str, int, bool)  # graph_name, window, enabled

    def __init__(
        self,
        graphs: dict[str, dict],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize multi-graph widget.
        
        Args:
            graphs: Dictionary mapping graph names to config dicts.
                    Each config: {"x_label": str, "y_label": str}
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.graphs: dict[str, GraphWidget] = {}
        self._setup_ui(graphs)

    def _setup_ui(self, graphs: dict[str, dict]) -> None:
        """Setup UI with multiple graphs."""
        layout = QVBoxLayout(self)
        
        # Moving average controls
        ma_group = QGroupBox("Moving Averages")
        ma_layout = QHBoxLayout(ma_group)
        
        self.ma_checks: dict[int, QCheckBox] = {}
        for window in [10, 50, 100, 500]:
            cb = QCheckBox(f"MA{window}")
            cb.setStyleSheet("color: #e0e0e0;")
            cb.toggled.connect(lambda checked, w=window: self._on_ma_toggled(w, checked))
            self.ma_checks[window] = cb
            ma_layout.addWidget(cb)
        
        ma_layout.addStretch()
        layout.addWidget(ma_group)
        
        # Create graphs
        for name, config in graphs.items():
            graph = GraphWidget(
                title=name,
                x_label=config.get("x_label", "Episode"),
                y_label=config.get("y_label", "Value"),
            )
            self.graphs[name] = graph
            layout.addWidget(graph)

    def _on_ma_toggled(self, window: int, enabled: bool) -> None:
        """Handle MA checkbox toggle."""
        for graph in self.graphs.values():
            graph.set_moving_average(window, enabled)

    def add_point(self, graph_name: str, x: float, y: float) -> None:
        """Add point to a specific graph.
        
        Args:
            graph_name: Name of the graph.
            x: X coordinate.
            y: Y coordinate.
        """
        if graph_name in self.graphs:
            self.graphs[graph_name].add_point(x, y)

    def clear_all(self) -> None:
        """Clear all graphs."""
        for graph in self.graphs.values():
            graph.clear()
