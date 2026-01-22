"""Environment visualization panel with Lunar Lander rendering."""

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QSlider, QGroupBox, QFrame
)


class EnvironmentPanel(QWidget):
    """Panel for visualizing the Lunar Lander environment.
    
    Features:
    - Live environment rendering
    - Episode reward slider
    - Episode/timestep counters
    - Epsilon display
    - Multi-instance tabs
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the environment panel."""
        super().__init__(parent)
        
        self._setup_ui()
        
        # State
        self.current_frame: QPixmap | None = None
        self.episode_reward: float = 0.0
        self.episode: int = 0
        self.timestep: int = 0
        self.total_timestep: int = 0
        self.epsilon: float = 1.0

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Tab widget for multiple instances
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3a3a5a; }
            QTabBar::tab { padding: 8px 16px; }
        """)
        
        # Create primary instance tab
        primary_tab = self._create_instance_tab("Instance 1")
        self.tab_widget.addTab(primary_tab, "Instance 1")
        
        layout.addWidget(self.tab_widget)
        
        # Episode info panel
        info_group = QGroupBox("Episode Info")
        info_layout = QVBoxLayout(info_group)
        
        # Reward slider
        reward_layout = QHBoxLayout()
        reward_layout.addWidget(QLabel("-600"))
        
        self.reward_slider = QSlider(Qt.Orientation.Horizontal)
        self.reward_slider.setRange(-600, 600)
        self.reward_slider.setValue(0)
        self.reward_slider.setEnabled(False)
        self.reward_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: linear-gradient(to right, #ff4444, #888888, #44ff44);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff0000;
                width: 12px;
                height: 20px;
                margin: -6px 0;
                border-radius: 6px;
            }
        """)
        reward_layout.addWidget(self.reward_slider)
        reward_layout.addWidget(QLabel("600"))
        info_layout.addLayout(reward_layout)
        
        self.reward_label = QLabel("episode reward = 0.00")
        self.reward_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reward_label.setStyleSheet("color: #e0e0e0;")
        info_layout.addWidget(self.reward_label)
        
        # Statistics
        stats_layout = QVBoxLayout()
        
        self.episode_label = QLabel("episode 0")
        self.episode_label.setStyleSheet("color: #ffd700; font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.episode_label)
        
        self.timestep_label = QLabel("episode timestep 0")
        self.timestep_label.setStyleSheet("color: #ffd700; font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.timestep_label)
        
        self.total_timestep_label = QLabel("total timestep 0")
        self.total_timestep_label.setStyleSheet("color: #ffd700; font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.total_timestep_label)
        
        self.epsilon_label = QLabel("epsilon (ε) = 1.00")
        self.epsilon_label.setObjectName("epsilonLabel")
        self.epsilon_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.epsilon_label)
        
        info_layout.addLayout(stats_layout)
        layout.addWidget(info_group)

    def _create_instance_tab(self, name: str) -> QWidget:
        """Create a tab for an environment instance.
        
        Args:
            name: Instance name.
            
        Returns:
            Tab widget.
        """
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # Environment render area - larger fixed size for full visibility
        self.render_label = QLabel()
        self.render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_label.setMinimumSize(480, 320)  # Larger minimum
        self.render_label.setFixedHeight(320)  # Fixed height for consistency
        self.render_label.setScaledContents(False)  # We handle scaling manually
        self.render_label.setStyleSheet("""
            background-color: #000000;
            border: 2px solid #3a3a5a;
            border-radius: 4px;
            color: #888888;
            font-size: 14px;
        """)
        
        # Set placeholder text
        self.render_label.setText("Environment will render here\nPress Play to start")
        
        tab_layout.addWidget(self.render_label)
        
        return tab

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the environment frame.
        
        Args:
            frame: RGB frame array from gymnasium render.
        """
        if frame is None:
            return
        
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        
        # Convert numpy array to QImage
        q_image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.render_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.render_label.setPixmap(scaled)
        self.current_frame = scaled

    def update_stats(
        self,
        episode: int | None = None,
        timestep: int | None = None,
        total_timestep: int | None = None,
        epsilon: float | None = None,
        reward: float | None = None,
    ) -> None:
        """Update the statistics display.
        
        Args:
            episode: Current episode number.
            timestep: Current episode timestep.
            total_timestep: Total timesteps across all episodes.
            epsilon: Current exploration rate.
            reward: Current episode reward.
        """
        if episode is not None:
            self.episode = episode
            self.episode_label.setText(f"episode {episode}")
        
        if timestep is not None:
            self.timestep = timestep
            self.timestep_label.setText(f"episode timestep {timestep}")
        
        if total_timestep is not None:
            self.total_timestep = total_timestep
            self.total_timestep_label.setText(f"total timestep {total_timestep}")
        
        if epsilon is not None:
            self.epsilon = epsilon
            self.epsilon_label.setText(f"epsilon (ε) = {epsilon:.2f}")
        
        if reward is not None:
            self.episode_reward = reward
            self.reward_label.setText(f"episode reward = {reward:.2f}")
            # Clamp slider value
            slider_val = max(-600, min(600, int(reward)))
            self.reward_slider.setValue(slider_val)

    def add_instance(self, name: str) -> int:
        """Add a new environment instance tab.
        
        Args:
            name: Instance name.
            
        Returns:
            Index of the new tab.
        """
        tab = self._create_instance_tab(name)
        return self.tab_widget.addTab(tab, name)

    def remove_instance(self, index: int) -> None:
        """Remove an environment instance tab.
        
        Args:
            index: Tab index to remove.
        """
        if self.tab_widget.count() > 1:
            self.tab_widget.removeTab(index)

    def clear(self) -> None:
        """Clear the display."""
        self.render_label.clear()
        self.render_label.setText("Environment will render here\nPress Play to start")
        self.update_stats(episode=0, timestep=0, total_timestep=0, epsilon=1.0, reward=0.0)
