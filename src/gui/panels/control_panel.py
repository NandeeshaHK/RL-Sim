"""Training control panel with play/pause, speed, and algorithm selection."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QSlider,
    QSpinBox, QDoubleSpinBox, QFormLayout, QScrollArea,
    QFileDialog, QMessageBox
)


class ControlPanel(QWidget):
    """Control panel for training controls and hyperparameters.
    
    Signals:
        play_clicked: Emitted when play is clicked
        pause_clicked: Emitted when pause is clicked
        stop_clicked: Emitted when stop is clicked
        reset_clicked: Emitted when reset is clicked
        speed_changed: Emitted with new speed multiplier
        algorithm_changed: Emitted with new algorithm name
        save_clicked: Emitted when save is clicked
        load_clicked: Emitted when load is clicked
        parameter_changed: Emitted with (param_name, new_value)
    """

    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    speed_changed = pyqtSignal(int)  # Speed multiplier
    algorithm_changed = pyqtSignal(str)
    save_clicked = pyqtSignal()
    load_clicked = pyqtSignal()
    parameter_changed = pyqtSignal(str, object)
    add_instance_clicked = pyqtSignal()
    remove_instance_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the control panel."""
        super().__init__(parent)
        
        self.is_playing = False
        self.current_speed = 1
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # === Playback Controls ===
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)
        
        # Play/Pause/Stop buttons
        btn_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setObjectName("playButton")
        self.play_btn.clicked.connect(self._on_play_clicked)
        btn_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setObjectName("pauseButton")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        btn_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        btn_layout.addWidget(self.reset_btn)
        
        playback_layout.addLayout(btn_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "2x", "5x", "10x", "Max"])
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_combo)
        
        speed_layout.addStretch()
        playback_layout.addLayout(speed_layout)
        
        layout.addWidget(playback_group)
        
        # === Algorithm Selection ===
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout(algo_group)
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Double DQN", "Rainbow DQN"])
        self.algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(self.algo_combo)
        
        layout.addWidget(algo_group)
        
        # === Instance Management ===
        instance_group = QGroupBox("Instances")
        instance_layout = QHBoxLayout(instance_group)
        
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self.add_instance_clicked.emit)
        instance_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("- Remove")
        remove_btn.clicked.connect(self.remove_instance_clicked.emit)
        instance_layout.addWidget(remove_btn)
        
        layout.addWidget(instance_group)
        
        # === Hyperparameters ===
        params_group = QGroupBox("Hyperparameters")
        params_layout = QFormLayout(params_group)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 0.1)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.00001)
        self.lr_spin.valueChanged.connect(
            lambda v: self.parameter_changed.emit("learning_rate", v)
        )
        params_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Gamma
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.9, 0.9999)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setValue(0.99)
        self.gamma_spin.setSingleStep(0.001)
        self.gamma_spin.valueChanged.connect(
            lambda v: self.parameter_changed.emit("gamma", v)
        )
        params_layout.addRow("Gamma:", self.gamma_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 256)
        self.batch_spin.setValue(64)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.valueChanged.connect(
            lambda v: self.parameter_changed.emit("batch_size", v)
        )
        params_layout.addRow("Batch Size:", self.batch_spin)
        
        # Epsilon (for Double DQN)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setValue(1.0)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.valueChanged.connect(
            lambda v: self.parameter_changed.emit("epsilon", v)
        )
        params_layout.addRow("Epsilon:", self.epsilon_spin)
        
        layout.addWidget(params_group)
        
        # === Checkpoint Controls ===
        checkpoint_group = QGroupBox("Checkpoint")
        checkpoint_layout = QHBoxLayout(checkpoint_group)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self._on_save_clicked)
        checkpoint_layout.addWidget(save_btn)
        
        load_btn = QPushButton("📂 Load")
        load_btn.clicked.connect(self._on_load_clicked)
        checkpoint_layout.addWidget(load_btn)
        
        layout.addWidget(checkpoint_group)
        
        # Add stretch at bottom
        layout.addStretch()

    def _on_play_clicked(self) -> None:
        """Handle play button click."""
        self.is_playing = True
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.play_clicked.emit()

    def _on_pause_clicked(self) -> None:
        """Handle pause button click."""
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_clicked.emit()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_clicked.emit()

    def _on_reset_clicked(self) -> None:
        """Handle reset button click."""
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.reset_clicked.emit()

    def _on_speed_changed(self, text: str) -> None:
        """Handle speed selection change."""
        speed_map = {"1x": 1, "2x": 2, "5x": 5, "10x": 10, "Max": 0}
        self.current_speed = speed_map.get(text, 1)
        self.speed_changed.emit(self.current_speed)

    def _on_algorithm_changed(self, text: str) -> None:
        """Handle algorithm selection change."""
        algo_map = {
            "Double DQN": "double_dqn",
            "Rainbow DQN": "rainbow",
        }
        self.algorithm_changed.emit(algo_map.get(text, "double_dqn"))

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        self.save_clicked.emit()

    def _on_load_clicked(self) -> None:
        """Handle load button click."""
        self.load_clicked.emit()

    def update_epsilon(self, value: float) -> None:
        """Update the epsilon display without triggering signal.
        
        Args:
            value: New epsilon value.
        """
        self.epsilon_spin.blockSignals(True)
        self.epsilon_spin.setValue(value)
        self.epsilon_spin.blockSignals(False)

    def set_algorithm(self, algorithm: str) -> None:
        """Set the algorithm selection.
        
        Args:
            algorithm: Algorithm name ("double_dqn" or "rainbow").
        """
        name_map = {
            "double_dqn": "Double DQN",
            "rainbow": "Rainbow DQN",
        }
        self.algo_combo.setCurrentText(name_map.get(algorithm, "Double DQN"))
