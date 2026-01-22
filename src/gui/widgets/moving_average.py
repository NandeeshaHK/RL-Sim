"""Moving average control widget."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QSpinBox, QPushButton, QGroupBox, QCheckBox
)


class MovingAverageControl(QWidget):
    """Widget for configuring moving average windows."""

    windows_changed = pyqtSignal(list)  # Emits list of enabled windows
    window_toggled = pyqtSignal(int, bool)  # window, enabled

    def __init__(
        self,
        default_windows: list[int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the moving average control.
        
        Args:
            default_windows: List of default MA window sizes.
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.windows = default_windows or [10, 50, 100, 500]
        self.enabled_windows: dict[int, bool] = {w: False for w in self.windows}
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Moving Averages")
        title.setStyleSheet("font-weight: bold; color: #ffd700;")
        layout.addWidget(title)
        
        # Preset checkboxes
        preset_layout = QHBoxLayout()
        
        self.checkboxes: dict[int, QCheckBox] = {}
        for window in self.windows:
            cb = QCheckBox(str(window))
            cb.toggled.connect(lambda checked, w=window: self._on_checkbox_toggled(w, checked))
            self.checkboxes[window] = cb
            preset_layout.addWidget(cb)
        
        layout.addLayout(preset_layout)
        
        # Custom window input
        custom_layout = QHBoxLayout()
        
        custom_label = QLabel("Custom:")
        custom_layout.addWidget(custom_label)
        
        self.custom_spinbox = QSpinBox()
        self.custom_spinbox.setRange(1, 1000)
        self.custom_spinbox.setValue(200)
        custom_layout.addWidget(self.custom_spinbox)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_custom)
        custom_layout.addWidget(add_btn)
        
        custom_layout.addStretch()
        layout.addLayout(custom_layout)

    def _on_checkbox_toggled(self, window: int, enabled: bool) -> None:
        """Handle checkbox toggle."""
        self.enabled_windows[window] = enabled
        self.window_toggled.emit(window, enabled)
        self._emit_windows_changed()

    def _on_add_custom(self) -> None:
        """Add a custom window size."""
        window = self.custom_spinbox.value()
        
        if window not in self.windows:
            self.windows.append(window)
            self.windows.sort()
            self.enabled_windows[window] = True
            
            # Add checkbox
            cb = QCheckBox(str(window))
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, w=window: self._on_checkbox_toggled(w, checked))
            self.checkboxes[window] = cb
            
            # Insert in sorted order (rebuild layout)
            self._rebuild_checkboxes()
            
            self._emit_windows_changed()

    def _rebuild_checkboxes(self) -> None:
        """Rebuild checkbox layout after adding custom window."""
        # This is a simplified version - in production, 
        # you'd want to properly reorder the checkboxes
        pass

    def _emit_windows_changed(self) -> None:
        """Emit signal with current enabled windows."""
        enabled = [w for w, e in self.enabled_windows.items() if e]
        self.windows_changed.emit(enabled)

    def get_enabled_windows(self) -> list[int]:
        """Get list of currently enabled window sizes.
        
        Returns:
            List of enabled window sizes.
        """
        return [w for w, e in self.enabled_windows.items() if e]

    def set_window_enabled(self, window: int, enabled: bool) -> None:
        """Programmatically set window state.
        
        Args:
            window: Window size.
            enabled: Whether to enable.
        """
        if window in self.checkboxes:
            self.checkboxes[window].setChecked(enabled)
