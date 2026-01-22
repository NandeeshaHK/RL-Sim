"""Neural network visualization panel."""

import math
from typing import Sequence

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QPainterPath, QLinearGradient
)
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


# Lunar Lander state feature names
STATE_LABELS = [
    "position x",
    "position y", 
    "velocity x",
    "velocity y",
    "angle",
    "angular velocity",
    "right leg contact",
    "left leg contact",
]

# Action names for Lunar Lander
ACTION_LABELS = [
    "no engine",
    "right engine",
    "down engine",
    "left engine",
]


class NetworkPanel(QWidget):
    """Panel for visualizing neural network architecture and activations.
    
    Displays:
    - Input layer with state feature labels
    - Hidden layers with activation values
    - Output layer with action Q-values
    - Connection weights between layers
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the network panel."""
        super().__init__(parent)
        
        self.setMinimumSize(500, 400)
        
        # Network structure
        self.layer_sizes: list[int] = [8, 64, 64, 4]  # Default for Lunar Lander
        
        # Current values
        self.input_values: np.ndarray | None = None
        self.q_values: np.ndarray | None = None
        self.layer_activations: list[np.ndarray] = []
        self.best_action: int = -1
        
        # Visual settings
        self.node_radius = 12
        self.layer_spacing = 120
        self.node_spacing = 25
        
        # Colors
        self.bg_color = QColor("#0a0a1a")
        self.node_color = QColor("#4a4a6a")
        self.active_color = QColor("#ff4444")
        self.connection_color = QColor("#2a2a4a")
        self.text_color = QColor("#e0e0e0")
        self.highlight_color = QColor("#ffd700")

    def set_network_structure(self, layer_sizes: Sequence[int]) -> None:
        """Set the network layer sizes.
        
        Args:
            layer_sizes: List of node counts per layer.
        """
        self.layer_sizes = list(layer_sizes)
        self.update()

    def update_values(
        self,
        input_values: np.ndarray | None = None,
        q_values: np.ndarray | None = None,
        layer_activations: list[np.ndarray] | None = None,
    ) -> None:
        """Update the displayed values.
        
        Args:
            input_values: Current state input values.
            q_values: Output Q-values for each action.
            layer_activations: Activations for hidden layers.
        """
        self.input_values = input_values
        self.q_values = q_values
        
        if layer_activations is not None:
            self.layer_activations = layer_activations
        
        if q_values is not None:
            self.best_action = int(np.argmax(q_values))
        
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the network visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.bg_color)
        
        if not self.layer_sizes:
            return
        
        # Calculate positions
        width = self.width()
        height = self.height()
        
        # Calculate layer positions
        num_layers = len(self.layer_sizes)
        layer_x_positions = []
        
        total_width = (num_layers - 1) * self.layer_spacing
        start_x = (width - total_width) / 2
        
        for i in range(num_layers):
            layer_x_positions.append(start_x + i * self.layer_spacing)
        
        # Calculate node positions for each layer
        node_positions: list[list[QPointF]] = []
        
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            positions = []
            # Limit displayed nodes for large layers
            display_size = min(layer_size, 12)
            
            total_height = (display_size - 1) * self.node_spacing
            start_y = (height - total_height) / 2
            
            for node_idx in range(display_size):
                x = layer_x_positions[layer_idx]
                y = start_y + node_idx * self.node_spacing
                positions.append(QPointF(x, y))
            
            node_positions.append(positions)
        
        # Draw connections
        self._draw_connections(painter, node_positions)
        
        # Draw nodes
        self._draw_nodes(painter, node_positions)
        
        # Draw labels
        self._draw_labels(painter, node_positions, layer_x_positions)

    def _draw_connections(
        self,
        painter: QPainter,
        node_positions: list[list[QPointF]],
    ) -> None:
        """Draw connections between layers."""
        for layer_idx in range(len(node_positions) - 1):
            current_layer = node_positions[layer_idx]
            next_layer = node_positions[layer_idx + 1]
            
            for from_pos in current_layer:
                for to_pos in next_layer:
                    # Color based on connection weight (random for demo)
                    gray = 40 + np.random.randint(0, 30)
                    pen = QPen(QColor(gray, gray, gray + 20), 0.5)
                    painter.setPen(pen)
                    painter.drawLine(from_pos, to_pos)

    def _draw_nodes(
        self,
        painter: QPainter,
        node_positions: list[list[QPointF]],
    ) -> None:
        """Draw network nodes with activations."""
        for layer_idx, positions in enumerate(node_positions):
            for node_idx, pos in enumerate(positions):
                # Determine node color based on activation
                if layer_idx == 0 and self.input_values is not None:
                    # Input layer
                    if node_idx < len(self.input_values):
                        val = abs(self.input_values[node_idx])
                        intensity = min(255, int(val * 100))
                        color = QColor(intensity, 50, 50)
                    else:
                        color = self.node_color
                        
                elif layer_idx == len(node_positions) - 1 and self.q_values is not None:
                    # Output layer
                    if node_idx < len(self.q_values):
                        if node_idx == self.best_action:
                            color = self.highlight_color
                        else:
                            color = self.active_color
                    else:
                        color = self.node_color
                else:
                    # Hidden layers
                    color = self.node_color
                
                # Draw node
                painter.setPen(QPen(color.darker(150), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(
                    pos,
                    self.node_radius,
                    self.node_radius
                )

    def _draw_labels(
        self,
        painter: QPainter,
        node_positions: list[list[QPointF]],
        layer_x_positions: list[float],
    ) -> None:
        """Draw input/output labels."""
        font = QFont("Consolas", 9)
        painter.setFont(font)
        painter.setPen(QPen(self.text_color))
        
        # Input labels
        if node_positions and len(node_positions[0]) <= len(STATE_LABELS):
            for idx, pos in enumerate(node_positions[0]):
                if idx < len(STATE_LABELS):
                    label = STATE_LABELS[idx]
                    value = ""
                    if self.input_values is not None and idx < len(self.input_values):
                        value = f": {self.input_values[idx]:.2f}"
                    
                    text = f"{label}{value}"
                    painter.drawText(
                        int(pos.x() - 140),
                        int(pos.y() + 4),
                        text
                    )
        
        # Output labels (Q-values)
        if node_positions and len(node_positions) > 0:
            output_layer = node_positions[-1]
            for idx, pos in enumerate(output_layer):
                if idx < len(ACTION_LABELS):
                    label = ACTION_LABELS[idx]
                    value = ""
                    if self.q_values is not None and idx < len(self.q_values):
                        value = f": {self.q_values[idx]:.2f}"
                    
                    # Highlight best action
                    if idx == self.best_action:
                        painter.setPen(QPen(self.highlight_color))
                    else:
                        painter.setPen(QPen(self.text_color))
                    
                    text = f"{label}{value}"
                    painter.drawText(
                        int(pos.x() + 20),
                        int(pos.y() + 4),
                        text
                    )
        
        # Hidden layer size labels
        painter.setPen(QPen(QColor("#888888")))
        for layer_idx in range(1, len(self.layer_sizes) - 1):
            x = layer_x_positions[layer_idx]
            size = self.layer_sizes[layer_idx]
            painter.drawText(
                int(x - 10),
                30,
                str(size)
            )
