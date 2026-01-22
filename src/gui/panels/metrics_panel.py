"""Metrics visualization panel with real-time graphs."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QGroupBox, QSplitter
)

from src.gui.widgets.graph_widget import GraphWidget
from src.gui.widgets.moving_average import MovingAverageControl


class MetricsPanel(QWidget):
    """Panel displaying training metrics in real-time graphs.
    
    Graphs:
    - Episode Rewards with moving averages
    - Q-Values (average and max)
    - V-Values (state value estimates)
    - Training Loss
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the metrics panel."""
        super().__init__(parent)
        
        self._setup_ui()
        
        # Track episode count
        self.episode_count = 0

    def _setup_ui(self) -> None:
        """Setup the UI with multiple graphs."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Training Metrics")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffd700;")
        layout.addWidget(title)
        
        # Moving average controls
        self.ma_control = MovingAverageControl(
            default_windows=[10, 50, 100, 500]
        )
        self.ma_control.window_toggled.connect(self._on_ma_toggled)
        layout.addWidget(self.ma_control)
        
        # Tab widget for different graph views
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumHeight(250)  # Ensure readable graph height
        
        # === Rewards Tab ===
        rewards_tab = QWidget()
        rewards_layout = QVBoxLayout(rewards_tab)
        rewards_layout.setContentsMargins(5, 5, 5, 5)
        
        self.rewards_graph = GraphWidget(
            title="Average Total Reward Across Episodes",
            x_label="episode",
            y_label="total reward",
        )
        self.rewards_graph.setMinimumHeight(200)  # Minimum graph height
        rewards_layout.addWidget(self.rewards_graph)
        
        self.tab_widget.addTab(rewards_tab, "Rewards")
        
        # === Q-Values Tab ===
        qvalues_tab = QWidget()
        qvalues_layout = QVBoxLayout(qvalues_tab)
        
        self.q_mean_graph = GraphWidget(
            title="Average Q-Value",
            x_label="episode",
            y_label="Q-value",
        )
        qvalues_layout.addWidget(self.q_mean_graph)
        
        self.q_max_graph = GraphWidget(
            title="Max Q-Value",
            x_label="episode", 
            y_label="Q-value",
        )
        qvalues_layout.addWidget(self.q_max_graph)
        
        self.tab_widget.addTab(qvalues_tab, "Q-Values")
        
        # === V-Values Tab ===
        vvalues_tab = QWidget()
        vvalues_layout = QVBoxLayout(vvalues_tab)
        
        self.v_graph = GraphWidget(
            title="State Value V(s)",
            x_label="timestep",
            y_label="V-value",
        )
        vvalues_layout.addWidget(self.v_graph)
        
        self.tab_widget.addTab(vvalues_tab, "V-Values")
        
        # === Loss Tab ===
        loss_tab = QWidget()
        loss_layout = QVBoxLayout(loss_tab)
        
        self.loss_graph = GraphWidget(
            title="Training Loss",
            x_label="training step",
            y_label="loss",
        )
        loss_layout.addWidget(self.loss_graph)
        
        self.tab_widget.addTab(loss_tab, "Loss")
        
        layout.addWidget(self.tab_widget)

    def _on_ma_toggled(self, window: int, enabled: bool) -> None:
        """Handle moving average toggle."""
        for graph in [
            self.rewards_graph,
            self.q_mean_graph,
            self.q_max_graph,
            self.v_graph,
            self.loss_graph,
        ]:
            graph.set_moving_average(window, enabled)

    def update_episode_metrics(
        self,
        episode: int,
        reward: float,
        q_mean: float | None = None,
        q_max: float | None = None,
    ) -> None:
        """Update metrics at episode end.
        
        Args:
            episode: Episode number.
            reward: Total episode reward.
            q_mean: Average Q-value during episode.
            q_max: Maximum Q-value during episode.
        """
        self.episode_count = episode
        
        self.rewards_graph.add_point(episode, reward)
        
        if q_mean is not None:
            self.q_mean_graph.add_point(episode, q_mean)
        
        if q_max is not None:
            self.q_max_graph.add_point(episode, q_max)

    def update_step_metrics(
        self,
        timestep: int,
        v_value: float | None = None,
        loss: float | None = None,
    ) -> None:
        """Update metrics at each training step.
        
        Args:
            timestep: Current timestep.
            v_value: State value estimate.
            loss: Training loss.
        """
        if v_value is not None:
            self.v_graph.add_point(timestep, v_value)
        
        if loss is not None:
            self.loss_graph.add_point(timestep, loss)

    def clear_all(self) -> None:
        """Clear all graphs."""
        self.episode_count = 0
        self.rewards_graph.clear()
        self.q_mean_graph.clear()
        self.q_max_graph.clear()
        self.v_graph.clear()
        self.loss_graph.clear()
